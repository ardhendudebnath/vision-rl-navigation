# Project plan and status

Staged build from the roadmap, with what is actually done recorded against it.
Phases are numbered as in the roadmap's Section 3.

## Status

| Phase | Description | Status |
|---|---|---|
| 0 | Repo, environment, task definition, metrics, test suite in CI | **Done** |
| 1 | Classical baseline (A* + pure pursuit) with full map access | **Done** |
| 2 | Privileged RL (PPO on exact pose + ground-truth ranges) | **Done** — 1.5M steps, val SPL 0.894 |
| 2b | Robustness suite across held-out and shifted environments | **Done** — hypothesis falsified, see below |
| 2c | Domain randomisation over shifts (what 2b argues for) | **Done** — gap not closed, failure mode changed |
| 2d | Compute sweep to 4.0M steps (tests the "needs more training" excuse) | **Done** — rejected |
| 2e | Caution-vs-progress reward ablation (3 arms) | **Done** — rejected; one real SPL win |
| 2f | Lidar beam-count experiment (32 / 64 / 128) | **Done** — decisive on one seed |
| 2g | Multi-seed replication (4 seeds x 2 arms) | **Done** — **2f does not replicate** |
| 2h | Direct perception audit (no training, no seeds) | **Done** — reconciles 2f and 2g |
| 2i | 16 vs 64 beams, 6 seeds/arm, pre-registered | **Done** — **significant; perception confirmed** |
| 3a | Depth camera observation mode | **Done** |
| 3b | Depth camera vs lidar, 6 seeds/arm, pre-registered | **Done** — **FOV beats resolution** |
| 3c | FOV sweep (90/180/270/360 deg), 24 seeds, pre-registered | **Done** — **monotone trend, forecast held** |
| 3d | Decoupling FOV from sample count | **Done** — **coverage causal, samples inert** |
| 3e | RGB + CNN encoder, 6 seeds/arm, pre-registered | **Done** — **the pixels are the problem** |
| 3f | RGB compute sweep to 4.0M (closes 3e's confound) | **Done** — rejected; gap survives 2.7x compute |
| 3g | Moving obstacles absent from the map | **Done** — hypothesis falsified; parity, later narrowed by 5b-5d |
| 3h | Trained on movers + frame stacking (6 seeds x 2 arms) | **Done** — parity; stacking inert |
| 4 | Technical report ([`report.md`](report.md)) + demo video | **Done** |
| 5 | Real Nav2 over ROS 2 as one more actor, 2 passes x 6 conditions | **Done** — **baseline was conservative, not weak** |
| 5b | Nav2 on the dynamic conditions | **Done** — parity fails on `dynamic_dense` |
| 5c | Frozen-mover subtraction (movers parked, map still wrong) | **Done** — **the parity was the planner degrading** |
| 5d | Replanning churn: measured, then causally tested | **Done** — churn real but **not** the explanation; better baseline adopted |
| 5e | DWB rollout horizon sweep with frozen control | **Done** — commitment length ruled out; motion cost still unexplained |
| 5f | Faster movers (0.8-1.5 m/s), 2 arms x 6 seeds, pre-registered | **Done** — **stacking inert at 3x speed**; the 3h speed explanation is dead |
| 5g | Explicit velocity channel vs frame_stack=2, 6 seeds/arm | **Done** — velocity inert in every encoding; extraction explanation dead too |
| 5h | Indifference reward (collision 5 == timeout 5), 2 arms x 6 seeds | **Done** — **stacking works at 1:1 (+0.033, p = 0.019, 6/6)**; the reward was hiding it |
| 5i | Recurrence (RecurrentPPO, 256-unit LSTM), 2 arms x 6 seeds | **Done** — **worse everywhere (-0.068 fast, -0.078 slow)**; the control refuses the motion reading |
| 5j | Re-price angular resolution at 1:1, 2 arms x 6 seeds | **Done** — **the reward flips the sign (interaction -0.142, p = 0.0043)**; 3d's null was success-only |
| 5k | Timeout audit of all six remaining seed_analysis results | **Done** — all reproduce exactly; **coverage deficits stall, sensing deficits crash** |
| 5l | Re-price coverage at 1:1, 6 new depth arms | **Done** — **headline survives (+0.080, p = 0.032)**; mechanism moves stall -> crash |
| 5m | Oracle motion prediction for the classical planner, 4 horizons | **Done** — **recovers 44% of the dense motion cost (p = 0.016)**; bounds better estimates for this planner |
| 5n | Oracle prediction for the initial plan and the slow-down | **Done** — **recovers nothing further**; +0.090 remains, not an information problem within this architecture |
| 5o | Robot agility 0.75x-2x, with and without the oracle | **Done** — **dense remainder unchanged at 2x (+0.005, CI [-0.045, +0.055])**; not a physical limit |
| 5p | Space-time A* with a when-blind ablation, 200 episodes | **Done** — **timing +0.050 on dense (p = 0.031), −0.050 on sparse (p = 0.002)**; access for robustness |
| 5q | Temporal safety margin 0-4 steps on the space-time agent | **Done** — **FIXED: sparse harm removed (+0.050, p = 0.002)**; motion cost 0.010 dense given oracle trajectories |
| 5r | Constant-velocity estimate in place of the oracle, 200 episodes | **Done** — **COSTLY: −0.065 on dense (p = 0.001), all collisions**; a 4-step margin recovers none |
| 5s | Re-price the encoder cost at 1:1, 12 new runs | **Done** — **SURVIVES (−0.180 narrow, p = 0.002)**; the failure stays in timeouts, unlike coverage's |
| 5t | Withhold the zero-margin fallback from the estimating agent | **Done** — **SYMPTOM: +0.010 on dense, bounded null**; the robot gets close before it falls back |
| 5u | Replan when an observation contradicts the estimate, 0.05 and 0.02 m | **Done** — **UNRESOLVED: +0.015 on dense, CI [−0.015, +0.050]**; sparse matches the oracle |
| 5v | Pixel stall audit of the Phase 5s policies, no training | **Done** — **information held (≤ 0.114 m)**; RGB stalls facing open routes its image features see |
| 5w | Cap the estimate's extrapolation at 2 s and 1 s | **Done** — **UNRESOLVED: +0.010 on dense**; a 1 s cap costs sparse 0.085 (p = 0.0005) |
| 6i | A mapper for dense scans: corroborate a cell's returns before believing them | **Done** — **UNRESOLVED at p = 0.0574**, but +0.140 (p = 0.0005) on the arm the stack actually runs |
| 6h | Fill the missing cell: this stack, without map or pose, at 360 beams | **Done** — **IMPLEMENTATION (+0.112, +0.120)**: the sensor fixes the pose, not the map |
| 6g | Ask a production stack the same question: Nav2 with slam_toolbox, no map, no pose | **Done** — **UNRESOLVED at 32 beams, IMPLEMENTATION at 360**: the same sensor, the same cost; a dense one, a seventh |
| 6f | Take away the pose: wheel odometry, and scan matching against the robot's own map | **Done** — **COSTLY, and matching PAYS**: free in clutter (−0.010, +0.040), ruinous in the open (−0.460 `large`) |
| 6e | The repaired mapping stack, re-run | **Done** — **COSTLY: −0.240 dense, −0.260 narrow**; Result 1's clutter margin was the map |
| 6d | The classical planner builds its own map (registered run) | **Recorded** — **COSTLY, from two bugs of mine**: every collision was into an obstacle mapped ≥1 s earlier |
| 6c | Observe only what a sensor could see: 360° scanner and 90° camera | **Done** — **UNRESOLVED for the scanner (−0.035)**; the camera costs −0.125 (p < 0.0001) |
| 6b | Where a released pixel policy crashes; probe for gap width | **Done** — **MIXED on where**; the CNN features cannot measure a gap (R² −0.29 against 0.22) |
| 6a | Fit the orbit by least squares instead of differencing | **Done** — **FILTER: +0.145 on dense at 1 cm noise (p < 0.0001)**; no longer distinguishable from the oracle |
| 5z | Gaussian noise on each observed mover position, 2 mm to 5 cm | **Done** — **UNRESOLVED at 1 cm (+0.005)**; noise costs more than the model bought; the fit wins at 5 cm |
| 5y | Fit each mover's oscillation instead of a straight line | **Done** — **MODEL: +0.070 on dense (p = 0.0001)**; matches the oracle, motion cost 0.005 |
| 5x | Replay with the velocity input overwritten, open and closed loop | **Done** — **PARTIAL: one-way latch (+0.367 exit, +0.008 entry)**; released, stalls become crashes |
| 6 | *(Stretch)* Isaac Lab / Habitat port, sim-to-real on hardware | Not started |

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

### Next experiment (what the result argued for)

**Domain randomisation over the shifts during training.** If the `dense` and
`narrow` collapse is OOD brittleness rather than a ceiling on reactive
control, sampling world configs across the shift distribution should close
most of it. Until that runs, "RL loses" is a claim about this training regime,
not about learned navigation — and the distinction is exactly what a committee
will probe. Run in Phase 2c below.

## Phase 2c — Domain randomisation: the gap did not close

A second policy, identical in every respect except its training worlds, which
were randomised per seed over arena size (10-16 m), obstacle count (1-22
circles, 0-10 boxes), obstacle size and start-goal separation — ranges chosen
to **contain** every evaluation shift. Same 1.5M-step budget.

Success rate / SPL, same 100 held-out worlds:

| Condition | Classical | PPO (nominal) | PPO (domain-rand.) |
|---|---|---|---|
| nominal | **1.000** / 0.985 | 0.960 / 0.910 | 0.940 / 0.865 |
| sparse | **1.000** / 1.000 | 0.980 / 0.963 | 0.990 / 0.951 |
| large | **1.000** / 0.990 | 0.970 / 0.940 | 0.970 / 0.925 |
| noisy_lidar | 1.000 / 0.985 | 0.960 / 0.909 | 0.940 / 0.869 |
| dense | **0.890** / 0.841 | 0.640 / 0.593 | 0.660 / 0.593 |
| narrow | **0.850** / 0.795 | 0.600 / 0.556 | 0.630 / 0.563 |

Success rate barely moves: +0.02 on `dense`, +0.03 on `narrow`, both inside
the noise at n=100. SPL is flat to three decimals on `dense`. Randomisation
also costs a little in-distribution performance — the usual robustness tax —
dropping `nominal` SPL from 0.910 to 0.865.

### The finding is in the failure composition, not the success rate

| Condition | Policy | Collisions | Timeouts | Timeout progress along route |
|---|---|---|---|---|
| dense | nominal | 29 | 7 | 5.4 m of 11.6 m |
| dense | domain-rand. | **12** | **22** | **7.9 m of 11.6 m** |
| narrow | nominal | 27 | 13 | 4.4 m of 10.6 m |
| narrow | domain-rand. | **11** | **26** | **7.5 m of 11.7 m** |

Randomisation taught real obstacle avoidance: collisions more than halved, and
failing episodes now reach roughly two-thirds of the route instead of ~40%.
But the policy pays for that safety in speed — its timeouts average about
0.16 m/s against a 0.6 m/s cap, and it exhausts the 500-step budget still
around 7 m from the goal. For comparison the classical planner finishes
`dense` in 226 steps.

**It became cautious rather than capable.** Widening the training distribution
moved failures between buckets rather than into successes, so the
`dense`/`narrow` collapse is *not* simply distribution mismatch.

### Caveats governing this

- **The DR ranges were chosen to contain the evaluation shifts.** This tests
  whether widening the training distribution recovers the loss, not
  generalisation to unseen *kinds* of shift. Evaluation worlds are still
  held-out seeds, so it is not leakage in the memorisation sense, but the
  distribution is deliberately matched and the report must say so.
- **The DR policy is compute-limited; the nominal one is not.** The nominal
  run flatlined from ~900k steps. The DR run was still improving at 1.5M
  (0.74 -> 0.76 validation success), so this is a "not at this budget" result.
  Its validation numbers are also not comparable to the nominal run's, since
  each validates on its own training distribution — the benchmark above is the
  comparable measurement.
- **n = 100 per condition**, so success-rate differences under ~0.05 are not
  resolvable.

## Phase 2d — Compute sweep: the "needs more training" excuse, tested

Phase 2c was discounted on the grounds that the DR policy was still improving
at 1.5M. **That was wrong, and worth recording as a methodological lesson.**
The claim rested on two points of a 50-episode validation curve (0.74 -> 0.76)
whose run-to-run jitter is about +/-0.06 — i.e. it was noise read as a trend.

The run was resumed (see `train.resume_from`; resuming is equivalent here
because every schedule is constant, which `train.py` now verifies rather than
assumes) and carried to **4.0M steps, 2.7x the nominal policy's budget**.

Validation plateaus by ~1.75M and then drifts slightly down: best SPL 0.770,
ending at 0.651. **The compute explanation is rejected.**

### The result, stated properly

`narrow`, 100 held-out episodes:

| Policy | Successes | Collisions | Timeouts | Timeout progress |
|---|---|---|---|---|
| nominal, 1.5M | 60 | 27 | 13 | 4.4 m of 10.6 m, 0.09 m/s |
| DR, 1.5M | 63 | 11 | 26 | 7.5 m of 11.7 m, 0.15 m/s |
| DR, 4.0M | **64** | **2** | **34** | 7.1 m of 11.8 m, 0.14 m/s |

`dense` shows the same shape: collisions 29 -> 12 -> 8, timeouts 7 -> 22 -> 24,
successes 64 -> 66 -> 68.

Collisions on `narrow` fall to **2 in 100**. The policy has all but learned not
to crash. But of the 25 episodes that left the collision bucket, **21 became
timeouts and only 4 became successes** — a roughly 5:1 conversion into
stalling rather than completion. The stalled episodes crawl at 0.14 m/s
against a 0.6 m/s cap.

Neither more compute nor a wider training distribution buys completion. Both
buy safety. **The policy converges on caution**, and that is the headline
finding of the whole comparison — more informative than the raw success-rate
gap, and it is a claim supported by a two-axis sweep rather than a single run.

### Why this is a good result for a report

Every cheap excuse for a losing RL result has now been tested and eliminated:
not too little data (1000 worlds), not the wrong distribution (DR containing
the shifts), not too little compute (2.7x, to plateau), not a weak evaluation
(identical worlds, identical order, n=100, classical given every privilege).
What remains is a substantive claim about what this reward and this
observation actually optimise for.

## Phase 2e — Caution-vs-progress ablation: the reward explanation, tested

### First, why the policy stalls

Rather than assume, episode returns were measured directly for the 4.0M
policy on `narrow` (n=100):

| Outcome | n | Mean return |
|---|---|---|
| success | 64 | +41.20 |
| timeout | 34 | -2.06 |
| collision | 2 | -24.91 |

A collision costs a flat -20. Timing out for all 500 steps costs -5
(`step_penalty` 0.01 x 500). **Crashing is four times worse than stalling
forever**, and the measured 22.85-point gap confirms it end to end.

This reframes Phases 2c and 2d: they eliminated the wrong suspects. The policy
is not failing to learn — it learned the optimum of the reward it was given.
The stalling is the reward's stated preference.

### The ablation

Three arms, each isolating one caution term, each trained from scratch at 1.5M
on the DR distribution so all are comparable to the DR baseline. Training from
the DR checkpoint would have confounded the reward change with the inherited
policy.

| Arm | Change | Rationale |
|---|---|---|
| `abl_step` | `step_penalty` 0.01 -> 0.05 | A full timeout now costs 25, *exceeding* the 20 collision penalty — directly inverts the incentive |
| `abl_noprox` | `proximity_penalty` 0.15 -> 0.0 | Removes the smooth gradient pushing the robot away from obstacles |
| `abl_lowcoll` | `collision_penalty` 20 -> 5 | Makes crashing roughly comparable to stalling |

`narrow`, 100 held-out episodes:

| Policy | Success | Collisions | Timeouts |
|---|---|---|---|
| classical | **0.850** | 0.120 | 0.030 |
| DR baseline | 0.630 | 0.110 | 0.260 |
| `abl_noprox` | 0.660 | 0.100 | 0.240 |
| `abl_step` | 0.660 | 0.280 | 0.060 |
| `abl_lowcoll` | 0.560 | **0.440** | **0.000** |

`dense` shows the same shape: as caution falls, collisions rise 0.120 -> 0.250
-> 0.340 -> 0.460 while timeouts fall 0.220 -> 0.140 -> 0.060 -> 0.000.

### The result, tested properly

All actors run identical worlds in identical order, so the comparison is
**paired** — an unpaired test here discards the world-difficulty pairing and
badly overstates variance. Paired 95% CIs on per-episode differences vs the DR
baseline (n=100):

| Arm | Condition | Metric | Delta | 95% CI | Verdict |
|---|---|---|---|---|---|
| `abl_step` | nominal | SPL | **+0.066** | [+0.019, +0.113] | **significant** |
| `abl_step` | nominal | success | +0.010 | [-0.034, +0.054] | not significant |
| `abl_step` | narrow | success | +0.030 | [-0.046, +0.106] | not significant |
| `abl_noprox` | narrow | success | +0.030 | [-0.046, +0.106] | not significant |
| `abl_lowcoll` | nominal | success | **-0.070** | [-0.127, -0.013] | **significant (worse)** |

**No arm significantly improves success rate under clutter.** Success stays
pinned in a 0.56-0.66 band while failures move between the timeout and
collision buckets almost one-for-one. The caution terms control *which* failure
occurs, not *how many*. The reward balance is not the bottleneck.

### The one real win

`abl_step` improves `nominal` SPL by +0.066, the only significant gain in the
ablation. Success is unchanged, so this is not about caution: the baseline was
**dawdling even on episodes it won**, and charging more per step tidies the
paths. Worth adopting as the default; it does nothing for the clutter problem.

### What is left

Four explanations for the losing RL result have now been tested and
eliminated: insufficient data (1000 worlds), wrong training distribution (DR
containing the shifts), insufficient compute (2.7x, to plateau), and reward
balance (three-arm ablation with paired tests). What remains:

1. **Perception.** 32 beams over 360 degrees is one sample every 11.25 degrees.
   Run as Phase 2f below — this is where the bottleneck actually was.
2. **Step budget.** `max_episode_steps=500` was chosen for 12 m arenas; DR
   trains up to 16 m where `l*` reaches 17 m. Still unrun; cheap, and it should
   be ruled out before the stalling claim goes into a report.

## Phase 2f — Lidar beam count: the bottleneck, found

### The geometric argument

A planar lidar samples the world at fixed angular spacing, so the *linear* gap
between adjacent rays grows with range. A gap the robot could physically drive
through is invisible if it falls between two rays:

| Beams | Angular spacing | Ray gap at 3 m | Resolves a 0.44 m robot-width gap out to |
|---|---|---|---|
| 32 | 11.25 deg | 0.59 m | **2.24 m** |
| 64 | 5.62 deg | 0.29 m | 4.48 m |
| 128 | 2.81 deg | 0.15 m | 8.96 m |

At 32 beams the robot cannot reliably resolve a gap it would fit through
beyond 2.24 m — barely more than one body length of lookahead. In `narrow`
worlds, where the whole task is committing to a specific opening, that is a
plausible hard limit that no amount of reward tuning could fix.

### Result

Two arms, `n_beams` 64 and 128, everything else identical to the DR baseline,
trained from scratch at 1.5M (the observation dimension changes, so resuming
is impossible).

Success / SPL on 100 held-out worlds:

| Condition | classical | 32 beams | 64 beams | 128 beams |
|---|---|---|---|---|
| nominal | **1.000** / 0.985 | 0.940 / 0.865 | 0.960 / 0.905 | 0.930 / 0.877 |
| sparse | **1.000** / 1.000 | 0.990 / 0.951 | 1.000 / 0.970 | 0.980 / 0.945 |
| large | **1.000** / 0.990 | 0.970 / 0.925 | 0.990 / 0.951 | 0.980 / 0.934 |
| dense | **0.890** / 0.841 | 0.660 / 0.593 | 0.730 / 0.680 | 0.630 / 0.571 |
| narrow | **0.850** / 0.795 | 0.630 / 0.563 | 0.700 / 0.629 | 0.650 / 0.576 |

Paired 95% CIs vs the 32-beam baseline (identical worlds, n=100):

| Arm | Condition | Metric | Delta | 95% CI | Verdict |
|---|---|---|---|---|---|
| `beams64` | narrow | success | **+0.070** | [+0.006, +0.134] | **significant** |
| `beams64` | narrow | SPL | **+0.066** | [+0.006, +0.126] | **significant** |
| `beams64` | dense | SPL | **+0.086** | [+0.002, +0.171] | **significant** |
| `beams64` | nominal | SPL | **+0.040** | [+0.005, +0.074] | **significant** |
| `beams128` | narrow | success | +0.020 | [-0.054, +0.094] | not significant |
| 128 vs 64 | dense | SPL | **-0.108** | [-0.203, -0.014] | **significant (worse)** |

**This is the first statistically significant success-rate improvement under
clutter in the entire study.** Collisions on `narrow` also fall from 0.110 to
0.030. After compute, distribution and reward all failed to move success, the
sensor did — which is the answer the geometric argument predicted.

### More beams is not monotonically better

128 beams is no better than the 32-beam baseline anywhere, and significantly
worse than 64 on `dense` SPL. The plausible reading is that a 133-dimensional
observation is harder to learn from at a fixed 1.5M-step budget and fixed
(256, 256) network. That is a hypothesis, not a result — and testing it means
either more compute for the 128 arm or a wider network, both of which are
confounded with the beam count unless controlled.

## Phase 2g — Multi-seed replication: 2f does not survive

Every arm in Phases 2c-2f was a single training seed, with significance
measured by pairing over *episodes*. That pairing controls world difficulty
and correctly answers "do these two policies differ on these worlds" — but it
is silent on training-seed variance. Phase 2g measures that variance directly:
both beam arms retrained on seeds 1-3, joining the existing seed-0 runs for
**four seeds per arm**, with the seed as the unit of analysis and an exact
permutation test (see `scripts/seed_analysis.py`).

### The result

`narrow`, one value per training seed, 100 held-out worlds each:

| Arm | Per-seed success | Mean +/- sd |
|---|---|---|
| 32 beams | 0.63, 0.68, 0.66, 0.52 | 0.623 +/- 0.071 |
| 64 beams | 0.70, 0.58, 0.64, 0.74 | 0.665 +/- 0.070 |

| Condition | Metric | Delta | p (exact) | Verdict |
|---|---|---|---|---|
| narrow | success | +0.042 | 0.457 | not significant |
| narrow | SPL | +0.025 | 0.514 | not significant |
| narrow | collision | +0.003 | 1.000 | not significant |
| dense | success | +0.073 | 0.229 | not significant |
| dense | SPL | +0.069 | 0.171 | not significant |
| nominal | success | +0.007 | 0.800 | not significant |

**The seed-to-seed spread (+/-0.07) is larger than the effect (+0.042).** The
arms' ranges overlap almost entirely: 0.52-0.68 against 0.58-0.74.

### How the single-seed comparison went wrong

Phase 2f reported `narrow` collisions falling 0.110 -> 0.030 as evidence that
better perception prevented crashes. Across seeds the two arms are
**indistinguishable**: 0.077 +/- 0.025 versus 0.080 +/- 0.048. Seed 0 of the
32-beam arm happened to be its *worst* for collisions, and seed 0 of the
64-beam arm its *best*. The single-seed comparison measured that coincidence
and nothing else.

This is worth keeping in the record as the clearest illustration in the
project of why single-seed RL comparisons mislead, and why the episode-level
CIs in Phases 2c-2f — which were correctly computed — still supported a
conclusion they could not bear.

### A subtlety worth noting

Best-validation SPL *did* separate completely across seeds: every 64-beam seed
(0.787 +/- 0.009) above every 32-beam seed (0.743 +/- 0.014). That looks like
strong evidence and is not, for two reasons. It is a **maximum** over ~30
validation points during training, which both biases it upward and suppresses
its variance; and it is measured on the DR training distribution rather than
on the held-out benchmark conditions. When a selection statistic and a
held-out measurement disagree, the held-out measurement wins.

### What survives

- The direction is positive in all six condition x metric comparisons, and the
  validation separation is real. A small genuine effect is plausible. But the
  six deltas are not independent, so their consistency cannot be converted
  into a p-value.
- **The robust conclusion of the whole study is untouched**: the classical
  planner beats every learned policy on every condition, by margins far larger
  than the seed spread (0.850 vs 0.665 on `narrow`).
- Every other single-seed finding in Phases 2c-2f, including the `abl_step`
  SPL win, should now be treated as **unreplicated**.

### What would settle the perception question

The geometry still predicts an effect, so the experiment deserves a better
design rather than abandonment. Option 3 was run first and is Phase 2h below;
it changes what options 1 and 2 are worth.

1. **More seeds.** At 4-vs-4 the smallest reachable two-sided p is 2/70 =
   0.029; resolving +0.04 against a +/-0.07 spread needs roughly ten seeds per
   arm.
2. **A larger contrast.** 16 versus 64 beams is a much bigger manipulation
   than 32 versus 64.
3. **Measure the mechanism directly.** Cheapest and most informative: no
   training, no seed variance.

## Phase 2h — Direct perception audit: the mechanism, measured

Instead of inferring a perception limit from success rate, measure it. Across
**480 on-route poses in 60 `narrow` worlds**, for each pose: sweep 2048
ground-truth rays to find every *traversable direction* (a heading the robot's
disc can translate 3 m along without collision, checked against world geometry
rather than any sensor), group them into maximal gaps, and ask whether an
N-beam scan puts at least one sufficiently-long beam inside each gap. A gap
with no beam in it is invisible to any policy, however good.

Poses are sampled along the A* route rather than uniformly over free space,
because open floor the robot never crosses is not where perception matters.

### Result

| Beams | Spacing | Gaps | Detected | Rate |
|---|---|---|---|---|
| 16 | 22.50 deg | 1164 | 906 | 0.778 |
| 32 | 11.25 deg | 1164 | 1030 | **0.885** |
| 64 | 5.62 deg | 1164 | 1105 | **0.949** |
| 128 | 2.81 deg | 1164 | 1134 | 0.974 |

By gap angular width:

| Width | n | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| 0-5 deg | 132 | 0.129 | **0.258** | 0.553 | 0.773 |
| 5-10 deg | 122 | 0.451 | 0.738 | 1.000 | 1.000 |
| 10-20 deg | 203 | 0.635 | 0.980 | 1.000 | 1.000 |
| 20-45 deg | 372 | 0.995 | 1.000 | 1.000 | 1.000 |
| >45 deg | 335 | 1.000 | 1.000 | 1.000 | 1.000 |

**The deficit is real and sits exactly where the geometry predicted.** A
32-beam scan misses 11.5% of traversable gaps and **74% of gaps narrower than
5 degrees**. Everything wider than 20 degrees is seen by every sensor tested.

### This reconciles Phases 2f and 2g

Going 32 -> 64 beams recovers only **6.4 percentage points** of gap detection
(0.885 -> 0.949). An effect that small cannot clear a +/-0.07 seed spread in
success rate at four seeds per arm. So Phase 2f was not looking in the wrong
place — it was **underpowered**, and Phase 2h says by roughly how much rather
than leaving it a matter of opinion.

This is the most useful thing the audit does: it converts "the replication
failed, so who knows" into "the mechanism exists, is quantified, and is too
small for the experiment that was run".

### It also designs the next experiment

**16 vs 64 beams spans 0.778 -> 0.949 in detection — a 17-point contrast,
nearly three times the 6.4 points of 32 vs 64.** That is the manipulation most
likely to produce a success-rate difference that survives seed noise, and the
audit identified it without training a single policy. Combined with more seeds
per arm, that is the experiment worth running; 32 vs 64 with more seeds is
fighting its own noise for a 6-point mechanism. Run as Phase 2i.

## Phase 2i — 16 vs 64 beams, powered and pre-registered

Twelve runs, six seeds per arm. The analysis plan was fixed **before the runs
finished**: primary endpoint `narrow` success rate, seed as the unit, exact
permutation test; everything else secondary. At six per arm the permutation
floor is 2/C(12,6) = 0.0022, against 0.029 at four per arm.

### Result

Success rate, one value per training seed, 100 held-out worlds each:

| Condition | 16 beams | 64 beams | Delta | p (exact) |
|---|---|---|---|---|
| **narrow** (primary) | 0.597 +/- 0.054 | 0.682 +/- 0.060 | **+0.085** | **0.035** |
| dense | 0.580 +/- 0.035 | 0.695 +/- 0.040 | **+0.115** | **0.002** |
| nominal | 0.898 +/- 0.041 | 0.937 +/- 0.021 | +0.038 | 0.056 |

Per-seed `narrow` success:

- 16 beams: 0.67, 0.51, 0.60, 0.59, 0.58, 0.63
- 64 beams: 0.70, 0.58, 0.64, 0.74, 0.71, 0.72

On `dense` the arms separate completely — every 64-beam seed above every
16-beam seed — hence p at the floor. `dense` SPL is +0.083 (p = 0.004) and
`dense` collisions -0.087 (p = 0.039). `nominal` misses significance.

### The audit predicted this before the policies existed

Phase 2h measured a 17-point gap-detection contrast for 16 vs 64 and, by
scaling the +0.042 observed at 6.4 points, implied roughly +0.11 in success.
Observed: +0.085 on `narrow`, +0.115 on `dense`. A 2.7x manipulation produced
a 2.0x effect — close to linear in detection rate.

**A training-free measurement forecast the outcome of a twelve-run training
experiment.** That is considerably stronger evidence for the mechanism than
the effect size on its own, because it is a prediction rather than a fit.

### Multiplicity, stated rather than buried

`narrow` success was pre-registered as the single primary endpoint, so p =
0.035 stands uncorrected — that is precisely what pre-registration buys. The
eight secondary tests do require correction: at Bonferroni (0.05/8 = 0.006),
`dense` success and `dense` SPL survive; nothing else does.

This matters because p = 0.035 would **not** survive correction if `narrow`
were treated as one of nine exploratory tests. The only thing distinguishing
those two readings is having fixed the endpoint in advance — which is exactly
the discipline Phase 2f lacked.

### What this establishes, and what it does not

Established: **sensor angular resolution is a real constraint on this task**,
with a measured mechanism, a pre-registered confirmation, and a quantitative
prediction that held.

Not established:

- **64 beams is not sufficient.** The best 64-beam policy still reaches 0.682
  on `narrow` against the classical planner's 0.850. Perception was *a*
  bottleneck, not *the* bottleneck.
- **The shape of the relationship.** Three points (16, 32, 64) roughly
  consistent with linearity in detection rate is not a characterised curve,
  and 128 beams was worse than 64 on a single seed, which remains unexplained
  and unreplicated.
- **Transfer to the vision setting.** Whether an image encoder inherits the
  same angular-resolution constraint is the question Phase 3 should be
  designed around, and this result is the reason to design for it explicitly
  rather than discover it later.

### Method notes worth keeping

- The ground-truth sweep must out-resolve the sensor under test, or the
  "truth" inherits the sensor's own blind spots and detection is overstated.
  `audit_pose` refuses a reference coarser than 8x the sensor.
- Detection is monotone in beam count on real worlds; this is asserted in the
  tests, because a bug in the beam-to-gap indexing would silently break it and
  the numbers would still look plausible.
- Probe distance matters. At 1 m almost every direction is traversable, the
  audit degenerates to one all-encompassing gap, and the measurement says
  nothing. 3 m is long enough that obstacles actually partition the sweep.

## Phase 3a — Groundwork: getting off privileged observations

No experiment; the seam that made Phases 3b-3f possible. `NavEnvConfig.obs_mode`
selects the observation, and deliberately raises `NotImplementedError` for a
mode that is not built rather than silently returning something plausible —
the failure mode being guarded against is a run that trains happily on the
wrong observation and is only caught, if ever, at analysis time.

Planned order of work, all of which was subsequently done:

1. Egocentric depth strip from the existing ray-caster — the cheapest honest
   step away from privileged state, since it is the same geometry in a
   camera-shaped observation (Phase 3b).
2. Rendered egocentric RGB behind a CNN encoder (Phase 3e).
3. Frame stacking (Phase 3h).

Two later bug classes trace back to this seam and are worth recording here:
evaluation configs that did not carry `obs_mode` through, and a camera
mis-labelled as a lidar in the benchmark. Both were silent. They are why
`src/vision_nav/training/run_spec.py` exists, and why the benchmark now reports the sensor
each policy actually reads rather than the one its row is named after.

## Phase 3b — Depth camera vs lidar: field of view beats resolution

Phases 2f-2i established that sensor angular resolution is a real constraint.
The natural follow-up is *which* property of the sensor matters, and a
forward-facing depth camera against a 360-degree lidar isolates it, because
the two trade off in opposite directions:

| Sensor | Angular resolution | Resolves a 0.44 m gap to | World visible |
|---|---|---|---|
| 64 beams / 360 deg | 5.62 deg | 4.5 m | 100% |
| 64 columns / 90 deg | 1.41 deg | 17.9 m | 25% |

The camera is 4x finer per degree while seeing a quarter of the world. Both
produce a 69-dimensional observation for an identical (256, 256) network, so
capacity is matched and the sensor is the only difference. `nav_depth.yaml`
matches `nav_dr.yaml` in every other respect.

Six seeds per arm. Primary endpoint `narrow` success, pre-registered, with a
directional prediction recorded before the runs: **the camera loses**, on the
grounds that Phase 2h measured 64-beam gap detection already at 0.949, leaving
little for extra resolution to buy, against a large new constraint.

### Result

| Condition | Lidar 360 | Depth 90 | Delta | p (exact) |
|---|---|---|---|---|
| **narrow** (primary) | 0.682 +/- 0.060 | 0.585 +/- 0.037 | **-0.097** | **0.019** |
| dense | 0.695 +/- 0.040 | 0.565 +/- 0.053 | **-0.130** | **0.004** |
| nominal | 0.937 +/- 0.021 | 0.862 +/- 0.039 | **-0.075** | **0.004** |

Per-seed `narrow` success:

- lidar 360: 0.70, 0.58, 0.64, 0.74, 0.71, 0.72
- depth 90: 0.60, 0.65, 0.55, 0.55, 0.58, 0.58

**Field of view dominates angular resolution.** Quadrupling angular precision
does not come close to paying for losing three quarters of the view.

Two secondary observations that both point the same way:

- **The penalty grows with clutter**: -0.075 (nominal), -0.097 (narrow),
  -0.130 (dense). That is the signature of peripheral awareness being the
  scarce resource — the more obstacles, the more it costs not to see beside
  and behind.
- **Collisions are statistically unchanged** on narrow and dense (+0.012,
  +0.008, both n.s.); the camera policy times out more instead (0.298 vs
  0.213 on narrow). Same cautious-rather-than-capable failure mode as the
  lidar policies under clutter, simply reached more often.

### On the prediction having been right

The direction was recorded before the runs. That is worth something — the
Phase 2h saturation argument had genuine forecasting content — but a confirmed
prediction is weaker evidence than a surprising one, because it cannot rule
out reasoning fitted to an expected answer. The stronger claim from this
project remains Phase 2h -> 2i, where a training-free measurement predicted a
*quantitative* effect size that then held.

### Limits

This compares **one** camera configuration. A 180 or 270 degree camera, or one
with memory to accumulate views across time, might close the gap entirely; the
FOV-versus-resolution trade has two knobs and this samples a single point on
it. What is established is the direction, not a frontier.

The obvious follow-ups, in order of value:

1. **FOV sweep at fixed column count** (90 / 180 / 270 / 360 degrees). This
   turns a single point into a curve and directly answers how much view is
   enough. It also isolates FOV from resolution, which the current comparison
   deliberately confounds.
2. **Frame stacking or recurrence.** If the deficit is *missing* information,
   more view fixes it; if it is *forgetting* what was just seen, memory fixes
   it. These predict different outcomes and the experiment distinguishes them.

## Phase 3c — FOV sweep: a monotone curve, and a second forecast that held

Phase 3b compared one camera against one lidar, which establishes a direction
and not a frontier. Sweeping field of view at a **fixed sample count**
(90 / 180 / 270 / 360 degrees, 64 samples) turns that single point into a
curve, and lets the Phase 2h audit make a second forecast before any of the
policies existed.

Pre-registered: a monotone increase in `narrow` success with field of view,
tested by Spearman rho against a Monte Carlo permutation null, 24 seeds.

### Result

**rho = +0.508, p = 0.013.** Monotone, as predicted.

The stronger part is the audit's quantitative forecast. It predicted the two
*interior* levels — the ones it was not fitted on — to within **+0.021 and
+0.001**. The tooling reports a mean absolute error of 0.006 across all four
levels, but two of those were the anchors used to fit the slope, so the honest
out-of-sample figure is 0.011. Stated that way in the report rather than
quoting the flattering number.

Two independent forecasts from one training-free measurement (Phase 2i and
this) is the strongest evidence in the project that the perception mechanism
is real rather than fitted after the fact.

## Phase 3d — Coverage is causal, sample count is inert

"Field of view beats angular resolution" is the obvious reading of 3b and 3c,
and it is wrong in a way worth pinning down: it implies a frontier where
either knob buys performance. Resolution = FOV / samples, so the two cannot
both be held — but either can be, and each gives a different experiment.

Three predictions were pre-registered, two of them **nulls with magnitude
bounds**, which is the part that makes them falsifiable:

1. Doubling samples at fixed 90 degrees changes nothing, within +/-0.01.
2. Doubling samples at fixed 360 degrees changes nothing, within +/-0.01.
3. Quadrupling coverage at *identical* angular resolution produces the effect.

### Result

| Test | Contrast | Delta `narrow` | p |
|---|---|---|---|
| Samples at fixed FOV | 32 vs 64 @ 90 deg | +0.002 | 1.000 |
| Samples at fixed FOV | 64 vs 128 @ 360 deg | -0.003 | 0.955 |
| Coverage at fixed resolution | 32 @ 90 vs 128 @ 360, both 2.81 deg/sample | **+0.095** | **0.024** |

**All three held.** Doubling the sample count changes nothing, twice, at both
ends of the range; quadrupling coverage at identical resolution produces the
entire effect. There is no frontier — one knob is inert.

A null predicted with a magnitude bound and then observed at +0.002 and -0.003
is much better evidence than an unbounded "no significant difference", which
is compatible with any effect the design was too weak to see.

## Phase 3e — The encoder costs 0.16-0.24, and the prediction was wrong

Everything so far reads geometry as a vector. Real cameras deliver pixels. The
question is what the *representation* costs when the information is held
constant: the RGB camera renders the same geometry the depth camera measures —
same 90 degree FOV, same 64 columns, goal vector bit-identical between modes —
and only the encoding changes, pixels through a CNN rather than a vector
through an MLP.

Pre-registered prediction: **-0.05 to -0.10** on `narrow`.

### Result

**-0.218 on `narrow` (p = 0.002), 0.16-0.24 across every condition.** Every
depth seed beats every RGB seed. The encoder costs reliability too: seed
spread roughly doubles for success and quadruples for collisions.

The prediction failed by two to three times. Worth contrasting with 3c: the
forecasts that held were derived from a *measured* quantity; this one was
intuition expressed in the same confident register.

The render is clean — no texture, no lighting variation, no sensor artefacts —
so 0.16-0.24 is a **lower bound** on what pixels cost, not an estimate.

## Phase 3f — The compute confound, rejected

The obvious objection to 3e is that a CNN is harder to optimise and simply
needed more training. Two predictions were pre-registered: extra compute
narrows the gap by less than 0.05, and the gap remains significant.

### Result

At **2.7x the compute**, the gap is -0.162 (p = 0.030). **Both predictions
held.** The last standing alternative explanation for 3e is rejected: the
deficit is representational, not an optimisation artefact.

## Phase 3g — Moving obstacles: the map stops being correct

Every condition so far hands the classical planner a perfect, current, static
map — its largest privilege and the one real deployments lack. Obstacles that
move and are **absent from the map** are the first place a reactive policy has
a structural reason to win.

The asymmetry is load-bearing and is asserted in tests: movers are visible to
the range sensor and absent from the occupancy grid the planner reads. Leaking
either way would make the experiment meaningless.

Pre-registered: the zero-shot policies beat the planner where the map is
wrong.

### Result (as measured at the time)

| Condition | classical | learned (best) | delta | p |
|---|---|---|---|---|
| narrow (static) | 0.850 | 0.682 +/- 0.060 | -0.168 | 0.031 |
| dynamic | 0.870 | 0.860 +/- 0.033 | -0.010 | 0.625 |
| dynamic_dense | 0.750 | 0.710 +/- 0.042 | -0.040 | 0.125 |

**Falsified.** The policy does not win; the gap closes to statistical parity
and stops. A regime change, not a reversal.

### The near-miss that makes the rest of it trustworthy

The planner replans against a costmap containing the movers, because a
baseline driving blind into them would prove nothing. Sweeping that interval
showed replanning helps where movers exist (+0.06 to +0.07) and *hurts* where
they do not, dropping `narrow` from 0.850 to 0.690 through path churn in tight
corridors.

With one global setting, `narrow` would have read classical 0.690 against the
policy's 0.682 — 4 of 6 seeds above the baseline, a clean "parity in tight
corridors" claim that was **purely an artefact of a handicap introduced in the
name of fairness.** It survived only because the interval was swept rather
than assumed.

> **Superseded.** Phases 5b, 5c and 5d revise these numbers three times, each
> by improving the baseline rather than by new evidence about the policy. The
> `dynamic_dense` row in particular becomes -0.110 at p = 0.031 once the
> baseline replans sensibly, and the parity finding ends up surviving on
> `dynamic` alone. The report carries the current numbers; this entry records
> what was measured at the time.

## Phase 3h — Frame stacking is inert

If the policy loses under motion because it cannot perceive velocity, giving
it velocity should help. Frame stacking is the cheapest way to provide it, and
it is the one setup handing the policy information the map-based stack
structurally lacks.

Pre-registered: stacking beats its own unstacked control on the dynamic
conditions.

### Result

**-0.008 (p = 0.784) and +0.013 (p = 0.703).** Nothing, twice.

Three untested explanations, in the order they are worth testing: the movers
may be too slow (0.15-0.45 m/s against a 0.6 m/s robot) for anticipation to
pay; velocity may be present but hard to extract from raw stacked scans
without an explicit difference feature or recurrence; and the reward's 4:1
preference for stalling over crashing may suppress commitment even when
anticipation is possible.

Phase 5c later supplies a fourth reading that fits better than any of these:
the policy's apparent robustness under motion is not anticipation at all but
*absence of commitment* — it has no plan to invalidate — which predicts
exactly that extra velocity information buys nothing.

## Phase 4 — Packaging: report, demo video, and the harness behind both

`scripts/run_benchmark.py` evaluates every actor across six conditions:
in-distribution, four world-distribution shifts, and sensor noise. All actors
see identical worlds in identical order, asserted at evaluation time rather
than assumed, so rows are directly comparable. `BENCHMARK_CONDITIONS` lives in
the library rather than the script because a second runner — the Nav2 bridge
of Phase 5 — has to reproduce those conditions exactly, and two copies of that
table would drift.

The write-up is [`report.md`](report.md); a one-page distillation for an
application document is [`one_page_summary.md`](one_page_summary.md).

**The demo video needed its own correction.** Taking the first six seeds in
order gave six successes for both actors, which misrepresents a policy
measured at 0.70 on `narrow`. The clip is now stratified to match the measured
outcome rates: the learned policy succeeds in 4 of 6 and the classical planner
in 5 of 6, including one world where the *planner* is the one that crashes.
Every seed is listed in `scripts/make_comparison_video.py`, so the selection
is reproducible rather than flattering.

## Phase 5 — Nav2 as the baseline: the hand-written stack was the weak one

> This log skips Phases 3d–3g (FOV sweep, RGB encoder, RGB compute sweep,
> moving obstacles). Those are written up in
> [`report.md`](report.md), which is the current record.

The largest standing caveat was that `classical` is not Nav2 — structurally
analogous, but a hand-written script. Nav2 1.3.12 on ROS 2 Jazzy now runs as
one more actor through a ROS 2 bridge (`ros2_bridge/`): same worlds, same seed
order, same success criterion, same metrics, same normalised action space.

The expected outcome was a credibility upgrade with no new result. It was not.
Over two passes Nav2 is ahead on `narrow` by more than its run-to-run spread,
ahead on `dense` by less than it, and marginally behind on the four
uncluttered conditions. The report's `narrow` gap was understated. Full
numbers in §4.1 of the report.

The mechanism replicates more cleanly than the outcome: collisions fall
58–92% on `narrow` and 56–78% on `dense` in both passes, exactly as the
failure-mode counts in Section 4 predicted, but on `dense` the recovered
episodes become timeouts instead of successes. Fixing the diagnosed failure
mode is not the same as fixing the outcome, and reading only the success
column would have hidden that.

### The bug that reversed the answer

The first complete run said Nav2 was **worse** — by 0.130 on `dense`. That
result was reproducible, internally consistent, and wrong, and it erred in the
direction that flattered this project's own baseline.

Nav2's costmaps are stateful. Obstacle marks are cleared only by ray-tracing
from the robot's current pose, and between episodes the robot teleports into a
new world, so the global costmap accumulated obstacles from every previous
episode until NavFn could not find a route at all. Every other actor in the
study is evaluated on independent episodes; Nav2 was not.

What made it findable was a diagnostic that had nothing to do with the
outcome: counting episodes in which Nav2 issued *zero* velocity commands.
Those were 2% of `nominal`, 8% of `dense` and 12% of `large` — the arena with
the most area to pollute. A robot that never moves is not a navigation
failure, it is a harness failure, and success rate alone cannot tell them
apart. Clearing both costmaps at each reset took `large` from 12%
zero-command to 0/30 with 30/30 success.

`zero_command_episodes` is now reported on every run.

### A metric that measured the wrong thing

The first fairness check was `mean_commands_per_step`, intended to detect
Nav2's control loop being starved of wall-clock time by a simulator running
faster than real time. It does not measure that. It falls whenever Nav2 runs a
recovery behaviour, because `Wait` publishes no command at all, so it tracks
how often Nav2 got *stuck* — a property of the condition, not the machine. The
tell was a serial run on an idle box scoring *lower* than a contended parallel
one. It is kept as a diagnostic and no longer used as a validity gate.

### Nav2 is not deterministic

Two passes over identical worlds in identical order do not agree: the stack is
a set of asynchronous processes and timing jitter changes which trajectory DWB
selects. Every other actor here is either deterministic or replicated over six
training seeds. Nav2 is neither, so it is run repeatedly and reported as a
range, with a winner declared only when every pass falls the same side of the
noise band.

### What this establishes, and what it does not

Established: the hand-written baseline is not weak. It is competitive with a
production stack everywhere and conservative in tight corridors, so the
`narrow` gap in the report is a lower bound. The collision mechanism is the
one the report had already diagnosed from failure-mode counts — the third time
in this project that a measurement made for one purpose correctly predicted a
later experiment.

Not established: that Nav2 is better on `dense`. Both passes say so and the
collision reduction there is real, but the success margin sits inside the
run-to-run spread, and the rule that demotes it is the same rule Phase 2g
exists to enforce. Two passes bound Nav2's spread rather than estimating it.

## Phase 5b — The dynamic conditions, and half a retraction

The parity finding of Phase 3g/3h — the learned policy indistinguishable from
the planner once the map is wrong — is the project's most RL-favourable claim
and rests on the baseline Phase 5 just showed is conservative under clutter.
`dynamic_dense` is clutter *and* movers, so it was the claim most exposed.

Pre-registered prediction, extrapolating the measured collision reductions of
Phase 5: **Nav2 >= 0.92 on `dynamic`**. It reached 0.840-0.870. The prediction
failed and the pre-registered counter-hypothesis is what happened — DWB's
1.5 s horizon does not anticipate a crossing mover, so a mechanism that
repairs clutter does not transfer to motion. Nav2's collision rate on
`dynamic` is 0.130-0.160 against the baseline's 0.130: no better at all.

| Condition | hand-written | learned (best) | Nav2 | Nav2 - learned |
|---|---|---|---|---|
| dynamic | 0.870 | 0.860 +/- 0.033 | 0.840-0.870 | -0.020 to +0.010 |
| dynamic_dense | 0.750 | 0.710 +/- 0.042 | 0.860-0.880 | +0.150 to +0.170 |

On `dynamic` parity survives and is stronger for it: the policy matches *both*
classical stacks. On `dynamic_dense` it does not — Nav2 beats all six training
seeds. Nav2 wins there by removing the *clutter* collisions (0.250 to
0.110-0.140), not by handling the movers.

This is the second parity claim in the project to turn out to be about the
baseline. Phase 3g caught the first before publication by sweeping the replan
interval rather than assuming it; this one was published and needed a
different baseline to expose. The lesson generalises: **when a result favours
the thing being studied, the baseline is the first place to look.**

It also refines the calibration rule. Predictions from measurement had held
twice and intuition had failed twice; this one came from a measurement and
still failed, because it extrapolated across a change of mechanism the
measurement never spanned. Measurement-derived predictions hold *within* the
regime measured and revert to intuition outside it.

## Phase 5c — The frozen-mover subtraction: the parity was the planner

Phases 3g and 5b compared actors *within* a condition, which cannot say
whether parity means the policy coped or the planner broke. So each dynamic
condition was re-run with the movers parked at their t = 0 positions:
identical worlds and seeds, movers still absent from the map, only the motion
removed.

Numbers below are the final ones, with the baseline given the best replanning
policy Phase 5d went on to find. Measured first against the timed baseline,
the classical column read 0.980 / 0.870 (sparse) and 0.910 / 0.750 (dense);
the *cost of motion*, which is what this phase is about, barely moved.

| clutter | actor | frozen | moving | cost of motion |
|---|---|---|---|---|
| sparse | classical | 1.000 | 0.880 | -0.120 |
| | Nav2 | 0.985 | 0.855 | -0.130 |
| | learned (best) | 0.900 | 0.852 | **-0.048** |
| dense | classical | 0.980 | 0.820 | -0.160 |
| | Nav2 | 0.945 | 0.870 | -0.075 |
| | learned (best) | 0.778 | 0.710 | **-0.068** |

Frozen, the classical advantage returns and widens sharply: -0.100 (sparse)
and -0.202 (dense), 0 of 6 seeds above the baseline in both, p = 0.031,
against -0.020 and -0.110 with the movers running. **The parity was the
planner degrading, not the policy coping.** The learned policy is worse in
both regimes; it just degrades less, because a policy that never commits to a
path has nothing to invalidate when the world moves. Robustness by absence of
commitment, which is consistent with frame stacking having been inert.

### Two methodological notes

**Freezing had to happen after generation.** Zeroing `dynamic_amplitude` in
the world config looks equivalent and is not: mover placement validates the
swept path, so a zero sweep accepts positions the moving config rejects and
the worlds end up with different obstacles. The first attempt did exactly
that, and the geometry-identity test written alongside it caught the extra
mover immediately. The subtraction now asserts identical circles, boxes,
start, goal, mover centres and `l*` per seed.

**The baseline's replan interval was swept again, not inherited.** It moves
the frozen numbers by ~0.15 (0.750 to 0.910 on dense, 0.840 to 0.980 on
sparse between `replan=0` and `replan=10`). Assuming it would have understated
the baseline in exactly the direction that flatters the learned side, which is
the same trap Phase 3g documented. Phase 5d then found a setting better than
either — replan when blocked rather than on a timer — taking those two cells
to 0.980 and 1.000, so even the swept value was not the best available.

### It also corrected Phase 5b

Phase 5b reasoned from `dynamic` collision rates that Nav2's `dynamic_dense`
win must be clutter handling rather than motion handling. The subtraction says
the opposite: Nav2 leads the hand-written planner by +0.035 frozen and +0.120
moving, so most of that advantage appears only when things move. On sparse the
same comparison is +0.005 and -0.015, nothing either way.

A better local planner therefore buys motion robustness **only under clutter**.
The leading explanation, consistent with every number here but not isolated:
in open worlds there is room to route around a mover and replanning achieves
that, so both classical stacks perform alike; in tight worlds the robot must
dodge inside the corridor, and re-committing to a fresh global plan every
second is actively harmful -- the churn pathology Phase 3g measured costing
`narrow` 0.160. `dynamic_dense` forces the hand-written baseline into exactly
that combination. Logging path changes per episode would confirm or kill it.

## Phase 5d — Churn: measured, causal, and the wrong explanation

Phase 5c blamed the hand-written baseline's motion cost on path churn -- it
re-commits to a fresh global plan every second, and Phase 3g measured that
pathology costing `narrow` 0.160 in a purely static world. The explanation fit
every number available and was labelled a hypothesis. This tests it.

Churn is operationalised as the shift in the lookahead point the controller is
steering at, measured across a replan from an unchanged pose: how far the
commitment moves when the plan is rebuilt.

### It behaves exactly as the hypothesis requires

| cell | success | churn (m) | replans |
|---|---|---|---|
| sparse frozen | 0.980 | 0.026 | 15.5 |
| sparse moving | 0.870 | 0.037 | 14.3 |
| dense frozen | 0.910 | 0.038 | 19.5 |
| dense moving | 0.750 | **0.058** | 17.6 |

Highest where the motion cost is largest; not merely a function of replan
count, since `dense frozen` replans most and churns less; and within every
cell the episodes that collided churned more (+0.072 to +0.098).

All consistent, none of it a test.

### The intervention, with a control

`replan_on_block` rebuilds the plan only when a mover actually obstructs it,
cutting replans from ~17 per episode to ~1. Pre-registered: `dynamic_dense`
recovers by >= +0.05, frozen cells stay within +/-0.03.

| cell | timed | block-triggered | delta |
|---|---|---|---|
| sparse moving | 0.870 | 0.880 | +0.010 |
| sparse frozen | 0.980 | 1.000 | +0.020 |
| dense moving | 0.750 | **0.820** | **+0.070** |
| dense frozen | 0.910 | **0.980** | **+0.070** |

The treated cell moved exactly as predicted. **The control cell moved by the
same amount**, so the cost of motion is unchanged: -0.120 sparse and -0.160
dense, identical to the timed baseline. Churn is real and is caused by timed
replanning, but it is a *clutter* pathology with nothing to do with whether
obstacles move. **The hypothesis is falsified as an explanation of the motion
cost.**

Had the control been left out, +0.070 on `dynamic_dense` would have read as
clean confirmation. It cost one extra condition to run.

### Consequences

The motion cost stands unexplained. Ruled out: churn, planning failure (A*
never fails to find a route) and sensing (the movers are fully visible to the
baseline's costmap). The remaining candidate is that committing to any plan is
itself the cost -- an architectural property, consistent with its surviving
every configuration change tried.

Block-triggered replanning is a strictly better baseline: it wins on all four
dynamic cells and is identical to the previous best on the six static ones,
where a correct map means the path is never blocked and it never fires. Every
dynamic number in the report now uses it, which moves the published comparison
*against* the learned policy:

| condition | classical | learned (best) | delta | sign p |
|---|---|---|---|---|
| dynamic | 0.880 | 0.860 | -0.020 | 0.219 |
| dynamic_dense | 0.820 | 0.710 | **-0.110** | **0.031** |
| dynamic_frozen | 1.000 | 0.900 | -0.100 | 0.031 |
| dynamic_dense_frozen | 0.980 | 0.778 | -0.202 | 0.031 |

`dynamic_dense` was -0.040 and not significant against the timed baseline. The
parity finding now survives on `dynamic` alone, its third narrowing in three
phases -- and each narrowing came from improving the baseline, never from new
evidence about the policy.

## Phase 5e — Commitment length: the third mechanism, also ruled out

Phase 5d eliminated churn and left one candidate for the motion cost:
**committing to a plan is itself the cost**, since any trajectory is computed
against a snapshot and goes stale the moment the world moves. DWB's rollout
horizon (`sim_time`) is exactly the length of that commitment, so sweeping it
tests the hypothesis directly. The frozen arm is the control again.

Pre-registered: the cost of motion rises monotonically with horizon, by at
least 0.04 across the range, while the frozen control stays within +/-0.04.

### Result

| horizon | moving | frozen (control) | cost of motion |
|---|---|---|---|
| 0.5 s | 0.780 | 0.830 | -0.050 |
| 1.0 s | 0.910 | 0.960 | -0.050 |
| 1.5 s | 0.870 | 0.960 | -0.090 |
| 3.0 s | 0.690 | 0.740 | -0.050 |

**Failed, and cleanly.** Across a 6x horizon range the cost of motion is flat
at -0.050 (range 0.040, inside the noise band) while the control swings by
0.220 -- an inverted U peaking near 1.0-1.5 s, driven by collisions rising to
0.280 moving and 0.220 frozen at 3.0 s. Horizon length strongly determines
navigation competence and does not touch robustness to motion at all.

The moving row alone (0.780, 0.910, 0.870, 0.690) traces a tidy optimum and
would have supported a story about staleness. The control shows it is ordinary
controller tuning. That is the second consecutive mechanism proposed for this
phenomenon, refuted by its own control cell.

### A tuning result, deliberately not adopted

At 1.0 s Nav2 reaches 0.910 on `dynamic_dense` against 0.870 at the default
1.5 s. That is one pass and close to Nav2's run-to-run spread, but more
importantly adopting it would be **tuning on the evaluation set**. The
hand-written baseline's replan sweep has the same character and is stated
openly for that reason; both move in the direction that strengthens a
classical actor, so both are conservative with respect to this project's
central claim. Nav2 stays at its shipped default, and the better setting is
recorded here rather than folded into the headline numbers.

### Where this leaves the motion cost

Four mechanisms eliminated: churn, planning failure, sensing, commitment
length. What survives is only the observation that Nav2's local layer halves
the cost under clutter (-0.075 against -0.160) for a reason none of them
explains. The next candidate with a checkable asymmetry is velocity in the
costmap -- every actor here treats each scan as a static snapshot and none can
tell an approaching mover from a receding one, which predicts a head-on
versus crossing difference that the current design cannot produce.

## Phase 5f — Faster movers: the speed explanation for 3h, tested and dead

Phase 3h found frame stacking inert and offered three untested reasons. The
leading one was that the movers are too slow to be worth anticipating:
0.15-0.45 m/s against a 0.6 m/s robot. `dynamic_fast` raises them to
0.8-1.5 m/s, so they now outrun the robot.

The manipulation is clean by construction. Speed is drawn *after* the mover
placement test, which depends only on centre, direction, amplitude and radius,
and `omega = speed / amplitude` is derived from it, so the accepted mover set
is identical seed for seed and only the angular rate changes.
`test_fast_matches_dynamic_geometry` asserts that, and a second test asserts
peak speeds land in band and average above the robot's cap.

Two arms trained from scratch on fast movers, `frame_stack` 1 and 4, six seeds
each, 1.5M steps, identical to the 3h arms otherwise. The slow condition is
kept as the control.

Pre-registered: stack4 - stack1 >= +0.05 on fast (p < 0.05) and within +/-0.03
on slow. Recorded at the time: this is an intuition-derived prediction and the
calibration record then stood at 2/2 for measurement-derived and 0/5 for
intuition-derived, so the base rate was against it.

### Result

| condition | stack1 | stack4 | delta | p (exact) |
|---|---|---|---|---|
| **fast** (primary) | 0.652 +/- 0.052 | 0.625 +/- 0.058 | **-0.027** | 0.442 |
| slow (control) | 0.802 +/- 0.027 | 0.778 +/- 0.059 | -0.023 | 0.502 |

**Inert at three times the speed**, and by almost exactly the amount it was
inert at the original speed. The speed explanation is dead.

Not a ceiling or floor artefact: the condition bites hard. Tripling mover speed
costs the classical planner 0.880 -> 0.750 (collisions 0.110 -> 0.250) and both
learned arms about 0.150. Everything gets meaningfully worse; velocity
information still does not help.

### What it leaves standing

Phase 5c's reading -- that the policy's robustness under motion is *absence of
commitment* rather than anticipation -- predicts exactly this: extra velocity
information should buy nothing at any speed. Two independent experiments
across a 3x speed ratio now agree with it. The frame-stacking null stops being
a caveat and becomes a finding.

Calibration after this phase: 2 of 2 measurement-derived predictions held, 0
of 6 intuition-derived ones did.

## Phase 5g — Explicit velocity: the extraction explanation, also dead

Phase 5f killed the speed explanation for the frame-stacking null but left one
reading standing: stacked scans carry velocity only *implicitly*, so perhaps
an MLP simply cannot recover it from raw ranges. `obs_velocity` hands the
policy the per-beam range delta directly. The comparison arm is
`frame_stack=2`, the same information at nearly the same width (133 against
138 dimensions at 64 beams, a 3.8% difference stated rather than engineered
away). The only substantive difference is who does the subtraction.

Six seeds per arm, slow condition kept as the control. This was a directional
expectation rather than a numbered pre-registration, and is not counted in the
calibration tally for that reason.

### Result

| arm | fast | slow (control) |
|---|---|---|
| no velocity (`frame_stack=1`) | 0.652 +/- 0.052 | 0.802 +/- 0.027 |
| implicit (`frame_stack=2`) | 0.637 +/- 0.043 | 0.785 +/- 0.060 |
| implicit (`frame_stack=4`) | 0.625 +/- 0.058 | 0.778 +/- 0.059 |
| **explicit delta channel** | **0.678 +/- 0.034** | **0.820 +/- 0.028** |

The explicit channel beats frame stacking by +0.042 on fast (p = 0.108) and
+0.035 on the slow control (p = 0.249). Neither is significant and the two are
the same size, so it is a mild preference for the encoding rather than
anything being used for anticipation. Against no velocity information at all
it is worth +0.026. All four arms sit within 0.053 of each other, against
per-arm seed spreads of 0.03-0.06.

**Velocity does not help this policy in any encoding at any speed tested.**
Both readings of the Phase 3h null are now dead, and Phase 5c's
absence-of-commitment account is the one left standing.

A weak secondary observation, offered as such: frame stacking is mildly
*monotonically worse* -- 0.652 > 0.637 > 0.625 on fast and 0.802 > 0.785 >
0.778 on slow, the same ordering both times. Consistent with the extra width
being a small net cost, but every gap is inside the seed spread and this is
not a result.

### The fourth recurrence, and a mechanism instead of a docstring

Scoring the velocity arms crashed: the env produced 69 dimensions where the
policy expected 133, because `env_overrides_for_run` did not carry
`obs_velocity`. That is the fourth time a field was added to `NavEnvConfig`
and not carried -- after `obs_mode`, the RGB camera and `frame_stack`, all
three of which run_spec.py's own docstring already listed. Three of the four
failed silently; only this one and `frame_stack` crashed.

A docstring asking the next person to remember is not a mechanism.
`OBSERVATION_FIELDS` and `CONDITION_FIELDS` now classify every field of
`NavEnvConfig`, and `test_run_spec_covers_every_env_field` fails if a new
field appears in neither. `max_goal_distance` was found unclassified in the
process: it scales the goal-distance observation, so a policy trained at one
value misreads another. Every config to date uses 20.0, so carrying it changes
nothing today and stops it mattering silently later.

## Phase 5h — The reward was hiding it: frame stacking works at 1:1

Phases 5f and 5g killed two of Phase 3h's three explanations for frame
stacking being inert. The third: the reward suppresses commitment. A collision
costs 20; a full 500-step timeout costs 0.01 x 500 = 5. Crashing is four times
worse than stalling and Phase 2e showed the policy optimising exactly that.
A policy that will not act on a prediction has no use for one.

`collision_penalty: 5` makes the two costs **exactly equal** -- indifference,
derived from the timeout cost rather than picked. Everything else matches
nav_dyn_fast. Two arms, frame_stack 1 and 4, six seeds each.

Pre-registered: stack4 - stack1 >= +0.05 under indifference against -0.027 at
4:1. Intuition-derived, with the 0-for-6 base rate stated at the time.

### Result

| reward | stack1 | stack4 | delta | p (exact) |
|---|---|---|---|---|
| 4:1 (collision 20) | 0.652 +/- 0.052 | 0.625 +/- 0.058 | -0.027 | 0.442 |
| **1:1 (collision 5)** | 0.660 +/- 0.019 | **0.693 +/- 0.020** | **+0.033** | **0.019** |
| 1:1, slow control | 0.808 +/- 0.021 | 0.807 +/- 0.048 | -0.002 | 1.000 |

**Frame stacking works once the reward stops punishing commitment.** Six of
six seeds, significant, and absent on the slow control. Swing between rewards
+0.060. The magnitude prediction failed (+0.033 against +0.05) while the
mechanism and its specificity held -- a different kind of miss from the
previous six.

### What actually changed

| arm | success | collisions | timeouts |
|---|---|---|---|
| 4:1 stack1 | 0.652 | 0.300 | 0.048 |
| 4:1 stack4 | 0.625 | **0.282** | **0.093** |
| 1:1 stack1 | 0.660 | 0.333 | 0.007 |
| 1:1 stack4 | **0.693** | **0.302** | 0.005 |

Stacking cuts collisions under *both* rewards, by 0.018 at 4:1 and 0.031 at
1:1. The velocity information was being used all along. What differs is what
it is spent on: at 4:1 the collision saving is more than swallowed by timeouts
nearly doubling and net success falls; at indifference the same saving becomes
successes.

**The reward does not decide whether the policy can anticipate. It decides
what anticipation is for.** That is Phase 2e one level up: the reward makes
the policy stall rather than get through, and it also converts additional
information into additional stalling. An observation channel is worth only
what the objective lets the policy do with it.

### Consequences

Phase 5c's absence-of-commitment account needed correcting rather than
confirming. It is real, but **caused by the reward rather than intrinsic to
the policy**. Across two rewards, three encodings and a 3x speed ratio that is
now supported rather than merely last standing.

It also revises how the Phase 3h null should be read: not "velocity
information is useless to this policy" but "this reward makes velocity
information useless", which is a claim about experimental design rather than
about learned navigation.

## Phase 5i — Recurrence: worse everywhere, and not about motion

The fourth and last way of supplying motion information, and the only one that
learns what to retain rather than being handed a fixed summary. PPO becomes
RecurrentPPO with a 256-unit LSTM for actor and critic; same worlds, same
seeds, same 1.5M-step budget, every other hyperparameter copied from the
baseline so the difference is algorithm-only.

Pre-registered: no effect, |delta| < 0.03 and p > 0.05, slow movers as control.
Amended before evaluation, on the strength of Phase 5h, to predict collisions
falling while success stayed flat.

| Condition | memoryless | recurrent | delta | p (exact) | seeds higher |
|---|---|---|---|---|---|
| fast movers (primary) | 0.652 | 0.583 | **-0.068** | **0.017** | 1/6 |
| slow movers (control) | 0.802 | 0.723 | **-0.078** | **0.004** | 0/6 |

Both predictions were wrong, and in a new direction: recurrence is
significantly **worse**, and collisions *rose* (0.300
to 0.323) alongside timeouts
(0.048 to 0.093).
It is not the Phase 5h pattern of trading crashes for stalls; the arm is worse
at both.

The control cell is what makes this readable. The harm is as large on slow
movers as on fast, so it is a general property of the arm rather than a failure
to anticipate. Fourth time a control has refused a mechanism claim in this
study, after churn, commitment length, and the stacking speed explanation.

### Two artefacts closed before the number was read

**Under-training.** RecurrentPPO is harder to optimise, so "memory does not
help" and "the LSTM had not finished" predict the same endpoint. Final third of
validation minus the third before it: baseline +0.026,
recurrent -0.008, both inside the +/-0.03
band measured across the four arms that have completed this budget.

**An LSTM ignoring its own memory.** Clearing recurrent state at every step
costs +0.315 success across 6/6 seeds, so the policy genuinely
depends on its history. That also prices a harness near-miss: carrying state
through evaluation had to be added for this experiment, and without it the arm
would have scored about 0.268
against 0.652 — a catastrophic-looking result that
would have been pure evaluation bug, pointing the same direction as the truth.

### Consequences

The experiment answers a narrower question than it was built to ask. It shows
that at a matched sample budget a recurrent policy is worse everywhere on this
task; it does not show that memory cannot help, because the LSTM ran on
hyperparameters chosen for an MLP. What it does settle is the cost: 19x the
wall clock for the same samples, which is the number to weigh before reaching
for recurrence on a task like this one.

Phase 5h remains the load-bearing result. Nothing here touches it.

### Calibration

Prediction 10, and it failed twice. Registered as a null on the strength of
three measured nulls -- 2-frame stacking, 4-frame stacking, the explicit
velocity channel -- then amended before any evaluation, on the strength of
Phase 5h, to predict collisions falling while success stayed flat. Recurrence
was significantly *worse* (-0.068, p = 0.017) and collisions *rose*.

Both versions reasoned carefully from real measurements, and both carried
those measurements across a boundary none of them spanned: from fixed
hand-designed windows to learned memory. That is the same crossing, in a
different form, as the Nav2 >= 0.92 prediction's static-to-moving extrapolation
in Phase 5b. Careful reasoning from a measurement did not extend its reach; it
only made the overreach harder to notice.

## Phase 5j — The reward prices perception too, and Phase 3d overstated a null

Phase 5h showed the reward decides what an observation *channel* is worth.
Every perception result in this project was measured under the same 4:1
reward, so each was a statement about what the objective let the policy do
with a sensor rather than about the sensor. This tests that on the sharpest
case, and re-reading Phase 3d's own result files first turned up a correction
that needed no new training at all.

### The correction

Phase 3d concluded **"doubling the sample count changes nothing, twice"**. For
one of the two contrasts that is not true:

| 64 -> 128 beams @ 360 deg, `narrow` | delta | p |
|---|---|---|
| success | -0.003 | 0.955 |
| collision | **+0.083** | **0.006** |
| timeout | -0.080 | 0.113 |

Collisions and timeouts move by equal and opposite amounts. The same number of
episodes fail; what changes is *how*. The pre-registered prediction was a
success-rate null with a +/-0.01 bound and it held -- the overstatement is in
the prose, which generalised "success did not move" into "nothing happened".

The other contrast (depth 32 -> 64 @ 90 deg) is clean on every metric and all
three conditions, so this is specific rather than a general property of adding
samples: collision -0.040
(p = 0.517), timeout
+0.038 (p = 0.294).

**The cause was a tool, not a slip.** `seed_analysis.py` recorded success, SPL
and collision and discarded the timeout rate, so the claim could never have
been checked by the script that produced it. It records timeouts now, and
re-running all three Phase 3d contrasts reproduced every published number
exactly -- the numbers were right, the reading of them was not.

### Coverage, better characterised

The same re-run explains *how* the coverage result pays, which was never
reported. 32 samples @ 90 deg against 128 @ 360 deg, identical 2.81 deg/sample:

| condition | success | collision | timeout |
|---|---|---|---|
| narrow | **+0.095** (p = 0.024) | +0.032 (p = 0.543) | **-0.127** (p = 0.004) |
| dense | **+0.088** (p = 0.028) | +0.020 (p = 0.667) | **-0.108** (p = 0.013) |

Coverage buys successes out of **timeouts**, with collisions unmoved. A robot
that cannot see behind itself does not crash more; it gets stuck more.

### The re-pricing

Two arms, six seeds, identical to `b64`/`b128` in every field except
`collision_penalty` 20 -> 5, which makes a crash cost exactly what a full
500-step timeout costs. Verified by diffing the saved configs: one field
differs from the 4:1 arms, and the two 1:1 arms differ only in `n_beams`.

| `narrow` | success | collision | timeout |
|---|---|---|---|
| 4:1, 64 beams | 0.682 | 0.105 | 0.213 |
| 4:1, 128 beams | 0.678 | 0.188 | 0.133 |
| 1:1, 64 beams | 0.598 | 0.390 | 0.012 |
| 1:1, 128 beams | 0.630 | 0.332 | 0.038 |

The quantity of interest is the **interaction** -- not "does resolution help"
but "does the reward decide what resolution does" -- and neither single-reward
comparison can express it. Seeds are index-paired across all four cells, so
the per-seed delta is well defined within each reward and the interaction is
an exact permutation test over those twelve numbers.

| metric | 4:1 | 1:1 | interaction | p |
|---|---|---|---|---|
| success | -0.003 | +0.032 | +0.035 | 0.262 |
| collision | +0.083 | -0.058 | **-0.142** | **0.0043** |
| timeout | -0.080 | +0.027 | **+0.107** | **0.0087** |

**The sign reverses.** Under 4:1 the extra beams raise collisions, on 6 of 6
seeds. Under 1:1 the same extra beams lower them, on 5 of 6. Success moves by
+0.035 and is not significant under either reward,
which is precisely why reading success alone found nothing.

Read behaviourally: when stalling is cheap, extra resolution is spent
attempting more passages and crashing on some of them. When stalling costs the
same as crashing and the policy must commit anyway, the same extra resolution
is spent getting through more safely. **The sensor did not change. What the
objective let the policy do with it did.**

### Two guards on the reading

The reward change is not free, and reporting only the interaction would hide
it. On `narrow` at 1:1 both arms do *worse* in absolute terms --
0.682 to 0.598
for 64 beams and 0.678 to
0.630 for 128 -- with collisions roughly
tripling as timeouts collapse to near zero. **Phase 5h's "1:1 is better" was
specific to moving obstacles and does not generalise to static clutter.**

And the effect is confined to where resolution has work to do. On `nominal`,
where the 4:1 policy already succeeds 0.927
of the time, the 1:1 resolution delta is
+0.000 on success and
-0.002 on collisions. That is the
control cell: a general property of the reward change would have moved it too.

### Consequences

Result 2 was stated for an observation *channel*. It holds for a perception
*parameter* as well, which is a wider claim than Phase 5h licensed on its own.
The report's perception findings stand as success-rate results -- coverage is
causal, resolution is inert in success -- but "resolution is inert" now needs
its qualifier, because what resolution does to *behaviour* is significant,
reward-dependent, and reverses sign.

### Calibration

Prediction 11 was the first in the record that crossed no boundary at all --
same arms, same seeds, same condition, derived from this contrast's own 4:1
breakdown -- and I said in advance that if it failed anyway, the calibration
rule was weaker than claimed.

Its *number* essentially held: +0.032 against a predicted |delta| < 0.03, not
significant, as predicted. Its *reasoning* was wrong. It argued that extra
resolution buys aggression rather than accuracy, since at 4:1 128 beams crashed
more at equal success, so the 1:1 collision delta should not improve. It
improved by 0.058, and the interaction is significant.

What made the result interpretable was the discriminator registered alongside
the prediction, which named the observation that would separate the two
mechanisms before either was seen. Without it, a number inside its bound would
have been reported as a confirmed model. **A right number is not a right model,
and only a pre-registered discriminator tells you which one you had.**

## Phase 5k — The audit completed: perception deficits make it stall, not crash

Phase 5j found `seed_analysis.py` discarding the timeout rate, which is how
Phase 3d came to report "doubling the sample count changes nothing" over a
significant collision shift. That tool produced six other published results.
Leaving those unchecked would have made the correction a lucky catch rather
than a fixed problem, so all six were re-run with the repaired tool.

**Every one reproduces its published success, SPL and collision numbers
exactly, to 0.00e+00.** The numbers were never wrong; one reading of them was.
No further overstatement turned up: Phase 3d was the only case.

Two nulls come out *stronger* than published. Phase 3f rejected the "the CNN
just needed more compute" objection on success alone; it is now null on all
three channels (+0.057,
-0.072,
+0.015, none significant),
which is a much harder thing to explain away. Phase 2g's non-replication is
likewise null on behaviour, not merely on the headline.

### What the recovered column shows

| Contrast | Condition | Success | Collision | Timeout |
|---|---|---|---|---|
| 16 -> 64 beams | `dense` | +0.115* (p 0.002) | -0.087* (p 0.039) | -0.028 (p 0.457) |
| 16 -> 64 beams | `narrow` | +0.085* (p 0.035) | -0.050 (p 0.175) | -0.035 (p 0.515) |
| 64 -> 128 beams | `narrow` | -0.003 (p 0.955) | +0.083* (p 0.006) | -0.080 (p 0.113) |
| 32 -> 64 px at 90 deg | `narrow` | +0.002 (p 1.000) | -0.040 (p 0.517) | +0.038 (p 0.294) |
| 90 deg depth -> 360 deg lidar | `narrow` | +0.095* (p 0.024) | +0.032 (p 0.543) | -0.127* (p 0.004) |
| 90 deg depth -> 360 deg lidar | `dense` | +0.088* (p 0.028) | +0.020 (p 0.667) | -0.108* (p 0.013) |
| lidar -> depth (FOV loss) | `dense` | -0.130* (p 0.004) | +0.008 (p 0.859) | +0.122* (p 0.004) |
| lidar -> depth (FOV loss) | `narrow` | -0.097* (p 0.019) | +0.012 (p 0.714) | +0.085 (p 0.097) |
| depth -> RGB (encoder) | `nominal` | -0.162* (p 0.002) | +0.047 (p 0.227) | +0.115* (p 0.011) |
| depth -> RGB (encoder) | `narrow` | -0.218* (p 0.002) | +0.105 (p 0.158) | +0.113 (p 0.214) |
| RGB at 2.7x compute | `narrow` | +0.057 (p 0.524) | -0.072 (p 0.333) | +0.015 (p 0.900) |
| 32 -> 64 beams (Phase 2g) | `narrow` | +0.042 (p 0.457) | +0.003 (p 1.000) | -0.045 (p 0.571) |

`*` marks p < 0.05, exact permutation, seed as the unit.

Read down the collision and timeout columns and the perception section
reorganises itself:

**Coverage deficits make the robot get stuck.** Both directions of the same
manipulation agree. Going from a 90 deg camera to a 360 deg lidar at matched
angular resolution cuts timeouts
-0.127 and
-0.108; going the
other way raises them +0.122.
Collisions never move in any of these cells and never approach significance,
on the same six seeds that make the timeout shift significant -- so this is a
contrast between channels within a comparison, not a power argument.

**Sensing deficits make it crash.** At the bottom of the range the story
inverts: 16 to 64 beams cuts collisions
-0.087 (p =
0.039) with timeouts flat.
Below adequacy the robot cannot see obstacles and hits them; above adequacy
more samples buy no competence, and Phase 5j shows the reward decides whether
they buy aggression or caution instead.

**A bad encoder does both**, and on `nominal` the stalling dominates
(+0.115, p =
0.011, against
+0.047 on collisions,
not significant). The CNN policy is not reckless with the same geometry; it is
indecisive with it.

### Consequences

Nothing in the report's success-rate conclusions changes. What changes is that
they now have a mechanism, and it is not the obvious one: the intuition that a
worse sensor means more crashes is right only at the bottom of the range. Over
most of it, a worse sensor means a robot that stops.

This also puts a floor under the original error. A tool that discarded one of
three outcome channels ran for the entire perception study; the claim it
misled was caught only because a later phase made that exact failure mode
salient. **The check that finds this class of error is re-reading old result
files after learning something new, and it is worth doing deliberately rather
than by luck.**

## Phase 5l — Coverage re-priced: the finding survives, the mechanism moves

Phase 5k found coverage's entire benefit sitting in the timeout channel, and
Phase 5j found indifference collapsing timeouts to near zero. Together those
say the report's strongest perception claim -- coverage is causal, +0.095 at
p = 0.024 -- had only ever been measured under a reward that leaves its
mechanism room to operate. Six depth arms at `collision_penalty` 5 close that;
the lidar half (`b128i`) already existed from 5j.

| `narrow` | success | collision | timeout |
|---|---|---|---|
| 4:1, 32 px @ 90 deg | 0.583 | 0.157 | 0.260 |
| 4:1, 128 beams @ 360 deg | 0.678 | 0.188 | 0.133 |
| 1:1, 32 px @ 90 deg | 0.550 | 0.433 | 0.017 |
| 1:1, 128 beams @ 360 deg | 0.630 | 0.332 | 0.038 |

Coverage still pays: +0.080
(p = 0.032) on `narrow` and
+0.107 (p = 0.004)
on `dense`, against +0.095 and
+0.088 at 4:1.

### The interaction

| metric | 4:1 | 1:1 | interaction | p |
|---|---|---|---|---|
| success | +0.095 | +0.080 | -0.015 | 0.7706 |
| collision | +0.032 | -0.102 | **-0.133** | **0.0238** |
| timeout | -0.127 | +0.022 | **+0.148** | **0.0022** |

`dense` agrees: success interaction +0.018
(p = 0.708), collision
-0.133
(p = 0.030), timeout
+0.115 (p = 0.009).

**The size of the effect is invariant and its mechanism is not.** The success
interaction is null on both conditions -- coverage buys the same amount under
either reward -- while the collision and timeout interactions are significant
on both. At 4:1 the narrow-FOV policy stalls; at 1:1, forced to commit, it
drives into what it cannot see. Same sensor deficit, same cost, different
failure.

### Consequences

The report's strongest perception claim survives re-pricing, which is worth
more than it sounds: angular resolution did not (Phase 5j), and coverage was
tested precisely because it looked equally exposed.

Phase 5k's summary sentence needs its qualifier, though. "Coverage deficits
make the robot get stuck; sensing deficits make it crash" was measured
entirely at 4:1. Getting stuck is what a coverage deficit looks like *under a
reward that prices stalling cheaply*. What a coverage deficit reliably costs is
success; how it spends that cost is the reward's decision.

### Calibration

Prediction 12 held on both halves: success in the registered +0.05 to +0.12
band, and the mechanism relocating with a collision delta at or below -0.05.
It was registered with **low** confidence, because it crossed the 4:1 -> 1:1
boundary and predictions 2 and 10 both died at exactly that kind of crossing.

So the rule as stated was wrong, and this is the counterexample that fixes it.
The failures extrapolated a magnitude into a regime where **nothing had been
measured**. Here Phase 5j had already measured the far side -- timeouts
collapse at 1:1 -- so the prediction was not extrapolation but deduction from
two measurements that jointly bracket the new cell: if the timeout channel is
closed and coverage still helps, it must help through another channel.
**What matters is whether some measurement covers the new regime, not whether
a boundary is crossed.** Given both sides, the crossing is interpolation.

## Phase 5m — Oracle motion prediction: half the motion cost, and a ceiling

Report Section 12 ranked velocity-blindness first among the explanations left
for the classical stack's motion cost. The adopted baseline was given exact
knowledge of each mover's swept region over H seconds, for both planning and
the replan trigger; the initial plan stays static-only so the arms differ only
in how they replan. Full treatment in `dynamic_obstacles.md`.

Guards: at H = 0 the committed churn experiment reproduces across 800 episodes
to the last bit, and the frozen cells are bit-identical at every horizon.

Result on `dynamic_dense`: **+0.070 at 2 s, recovering
44% of a 0.160 motion cost**, p =
0.0156 against a Bonferroni-corrected 0.0167, 7 episodes
won and 0 lost, via collisions -0.100.
Neither neighbouring horizon is significant. `dynamic` agrees in direction but
not significance (+0.060 at 4 s, p = 0.070).

**The ceiling is as important as the recovery, and was first overstated.** An
oracle bounds every real *estimate* for this planner, which plans in space
around swept regions. It does not bound a planner reasoning in space-time with
the same input. The first writeup said velocity explains "at most about half"
the cost outright; Phase 5n is what made the difference sharp.

Calibration, prediction 13: registered as low-confidence intuition. Magnitude
landed in its +0.03 to +0.08 band; the "long horizons hurt" shape held on dense
and failed on sparse, where the best horizon was the longest. A 0.05-wide band
catching the answer is not evidence the forecast was good.

Also caught in passing: a new test helper reused the name of an existing one
and silently rebound it, breaking three churn tests. Ruff's F811 flags only
redefinition of an *unused* name, so the full suite was the only guard.

## Phase 5n — Prediction everywhere else: the other half does not move

Phase 5m's oracle fed replanning and the replan trigger and recovered half the
dense motion cost. Two places still read movers as snapshots: the initial plan
and the controller's reactive slow-down. Each got the same 2 s oracle, measured
against 5m's best arm, whose remaining cost is +0.090 on both
conditions. Full treatment in `dynamic_obstacles.md`.

"The initial plan sees movers" is two interventions -- seeing them at all
changes frozen worlds, seeing their futures does not -- so it was tested as two
arms, and only the second has an identity control. Checks: the 5m arm
reproduces Phase 5m on every rate, both frozen identities hold, and the churn
experiment still reproduces across 800 episodes to the last bit.

| added to replanning prediction | `dynamic` | `dynamic_dense` |
|---|---|---|
| initial plan sees movers at all | -0.010 (1 won, 2 lost) | +0.010 (2 won, 1 lost) |
| initial plan sees their futures | +0.000 (1 won, 1 lost) | +0.000 (0 won, 0 lost) |
| slow-down reads their futures | -0.040 (1 won, 5 lost, p = 0.22) | +0.000 (1 won, 1 lost) |

**Nothing moves.** Oracle knowledge at every point this stack reads movers
recovers half the motion cost and no more. Predictive slow-down leans the wrong
way on `dynamic` -- collisions +0.040 -- but
not significantly.

This also forced a correction to Phase 5m's writeup, which said velocity
explains "at most about half" the motion cost. An oracle bounds the information,
not its use, and this planner uses a perfect input crudely. What is established
is narrower: better information does not close the gap *within this
architecture*. The remainder has two separable explanations -- space-time
planning with the same oracle, or physical limits on evading a correctly
anticipated mover, testable by raising speed and acceleration limits alone.

### Calibration

Prediction 14, and it inverts the usual pattern: the measurement-derived part
failed and the intuition parts held.

- **"initS identical to base2 on every episode" -- failed.** Derived from a
  diagnostic in which 0 of 8 episodes had a mover on the route at t = 0. Across
  400 episodes a few percent did, and outcomes differed by one to three episodes
  a cell. The derivation was sound and the claim built on it was not: 0 of 8
  cannot measure "never". By the rule of three it is compatible with a true rate
  near 37%, so predicting zero differences in 400 was overreach from the start.
- **Initial-plan prediction null -- held.** +0.000 on both conditions.
- **Slow-down +0.00 to +0.04, not significant -- half held.** Not significant
  anywhere, but -0.040 on `dynamic` is outside the band and of the wrong sign.
- **Overall, neither closes the remaining half -- held.**

The lesson is distinct from the earlier three. A measurement can be taken in
exactly the right regime and still not support a claim made at a resolution it
never had.

## Phase 5o — Agility: the dense remainder is not a physical limit

Two explanations survived 5n for the half of the motion cost oracle prediction
cannot recover: crude use of a perfect input by a planner that reasons only in
space, or a robot that cannot physically evade a mover it has anticipated. Robot
velocity and acceleration limits were scaled together from 0.75× to 2×, 200
episodes each, with and without the 2 s oracle. Full treatment in
`dynamic_obstacles.md`.

Before running: the dense movers peak at 0.15-0.45 m/s and the robot does 0.6.
It already outruns them, which made a physical limit the less likely answer.

Checks: 1× reproduces Phase 5m on its first 100 episodes; the frozen identity
holds at every speed; 0.75× on dense fails the pre-declared controller-validity
rule (frozen success down 0.035, from timeouts) and is not interpreted.

**Result.** On `dynamic_dense` the cost remaining with the oracle moves
+0.005 from 1× to 2×, 95% CI
[-0.045, +0.055], against the −0.085
that full removal would need. Doubled agility and halved exposure
(188 to 98 steps)
leave it unchanged; the interval excludes removing more than about half.
`dynamic` is inconclusive (-0.030,
[-0.085, +0.025]). The oracle's gain replicates at every
speed, and Phase 5m's p = 0.016 on 100 episodes is p = 0.004 on 200.

Not claimed: that the motion cost is insensitive to exposure (the interval on
the no-oracle cost is [-0.070, +0.050]), or that the planner is
therefore the cause -- that is what survives, not what is shown.

### Calibration

Prediction 15: "exposure, not a physical limit; the recovered fraction stays
flat, its 2× − 1× interval including zero; remaining cost rises at 0.75×".

- *Not a physical limit* -- held on dense, decisively; inconclusive on sparse.
- *Exposure shrinks both arms alike* -- not supported. No detectable shrinkage
  on either condition.
- *Recovered fraction flat, interval includes zero* -- satisfied, and worthless.
  The intervals were near ±0.4 and would have included zero whatever was true.
- *Remaining cost rises at 0.75×* -- on dense, significantly, but in the cell the
  validity rule excludes; not on sparse.

The registered null was unbounded: the form Phase 3d named as compatible with
any effect a design is too weak to see, and that the report lists among the
practices it avoids. The conclusion rests instead on an interval on remaining
cost, added after the first run -- the per-episode data needed for it had not
been kept -- and labelled post hoc in the script, the result file and every
document. The lesson is the oldest one in this record, repeated: a null has to
say how big an effect it rules out.

## Phase 5p — Space-time planning: timing buys access and costs robustness

The explanation left standing after 5o: a planner reasoning only in space uses a
perfect input crudely. Tested with a space-time A\* -- optimal against brute
force on 65 random worlds, each safety rule mutation-tested -- driven by an
agent that tracks a schedule. Full treatment in `dynamic_obstacles.md`.

The agent differs from the spatial baseline in its window, replan timer, schedule
tracking and step-minimising paths as well as in timing, so the test is an
ablation: the same agent shown each mover's window-union at every step. Checks:
the spatial arm reproduces Phase 5m; full and swept are bit-identical on frozen
worlds. The prediction and a null bounded at ±0.03 were committed (`fe3e75e`)
before the result file existed.

| full − swept | gain | p | won / lost | 95% CI | registered label | corrected |
|---|---|---|---|---|---|---|
| `dynamic_dense` | +0.050 | 0.031 | 14 / 4 | [+0.010, +0.090] | MATTERS | MATTERS |
| `dynamic` | −0.050 | 0.002 | 0 / 10 | [−0.085, −0.020] | inconclusive | **HARMS** |

Timing matters on dense clutter through fewer timeouts (−0.060),
not fewer collisions (+0.010). It harms sparse worlds through
collisions (+0.050). The swept agent alone closes the sparse
motion cost against the spatial baseline (+0.095 to
+0.000), so full-against-spatial would have credited
timing with work the architecture did. On dense clutter the full agent cuts the
remaining motion cost from +0.085 to
+0.025.

### Two things found along the way

**A planner exploit.** Movers vanished past the planning window, so the cheapest
route through a corridor a frozen mover blocked was to wait out the window and
drive through; the window slides at every replan, so the robot would wait
forever. Found by the agent planning 1.7 waits an episode on worlds where nothing
moves. Movers now persist past the window at their last position; frozen waits
0.0. I guessed it also caused the one frozen failure in the smoke test; frozen
success did not change, so it did not.

**A verdict that could not say "worse", written twice.** The registered rule
named MATTERS and a bounded INERT, and the code applying it labelled a −0.050 at
p = 0.002 with no episode won "inconclusive" -- the defect fixed in
`fast_movers_experiment.py` earlier, repeated in a new script. HARMS is added
after the run, labelled as post hoc, with tests.

### Calibration

Prediction 16: "full minus swept on dense +0.03 to +0.08, p < 0.05, carried by
fewer collisions; frozen identical." Magnitude, significance and the identity
held; the mechanism did not -- it was timeouts. Derived from a 20-episode smoke
test, where full showed no collisions and swept one; at 200 episodes that
reading of mechanism did not survive, which is the Phase 5n lesson about small
samples in a new form. Partial.

## Phase 5q — A temporal safety margin: the motion cost, explained

Phase 5p's untested cause for timing's sparse harm: the planner threads gaps one
0.24 s step ahead of a mover, so tracking lag costs a collision. Tested with a
margin of m plan steps either side of each mover's occupancy. Pre-registered in
`2c2953e`, before the result file existed, with the primary margin fixed at two
steps to match the tracker's 0.5 s lead. Full treatment in `dynamic_obstacles.md`.

Checks: margins 0 and 4 bit-identical on frozen worlds; margin 0 and swept
reproduce Phase 5p on every episode. Registered decision: **FIXED**.

- H1, margin 2 against margin 0 on sparse: +0.050, p = 0.002,
  10 won and 0 lost, collisions −0.050.
  Dose-response 0 -> 1 -> 2 steps: 0.945, 0.980,
  0.995, flat at 4.
- H2, margin 2 against swept on dense: +0.065, p = 0.004 -- the
  gain survives and grows.
- Failed: four steps did not erode the dense gain towards swept.

With the margin, the motion cost is 0.010 on dense clutter and 0.000 on
sparse worlds, against 0.145 and 0.140 for the spatial baseline --
with oracle trajectories throughout. (Margin-2 frozen success is not run: it
equals margin 0 by the frozen identity, verified at margin 4 and unit-tested.)

### Consequences

The motion cost the report called unexplained was a planning problem: a planner
that could not reason about time, and once it could, needed a margin for its own
tracking error. That is conditional on perfect prediction, so the practical
question moves to how good an *estimated* trajectory has to be.

### Calibration

Prediction 17. H1 held on magnitude (+0.050, at the top of its +0.03 to +0.05
band), significance and mechanism; H2 held; the registered decision was reached.
The secondary shape clause -- four steps eroding the dense gain -- failed.
Partial, by the same standard that counted prediction 13 partial for a failed
shape clause, though the primary held more completely here.

Worth noting against the calibration rule: the magnitude came from a measurement
in the regime tested, but the *mechanism* was a hypothesis nothing had measured,
the crossing the rule says to distrust, and it held. One case does not revise a
rule; it is recorded so a second can.

## Phase 5r — A constant-velocity estimate: about half survives

Phase 5q's result was conditional on exact mover trajectories. Here the oracle is
replaced by the simplest real estimate: constant velocity from the last two
positions the agent observed, noise-free. Pre-registered in `d428cc5`, before the
result file existed. Full treatment in `dynamic_obstacles.md`.

Checks: estimate and oracle bit-identical on frozen worlds; the oracle arm
reproduces Phase 5q on every episode. Registered decision: **COSTLY**.

- Estimate against oracle, dense: −0.065, p = 0.001, 1 won and
  14 lost, CI [−0.105, −0.030]; collisions +0.065,
  timeouts +0.000.
- Sparse: −0.020, p = 0.125, 0 won and 4 lost,
  CI [−0.040, −0.005] -- neither significant nor bounded-inert.
- Four-step margin against two, dense: +0.000, CI [−0.025, +0.030]
  -- recovers nothing.

Motion cost with the estimate: 0.075 dense, 0.020 sparse (oracle 0.010 and
0.000; spatial baseline without prediction 0.145 and 0.140).

Post hoc, by `estimate_diagnostic.py`: all 18 lost episodes end in contact with a
mover; the estimate was a median 0.030 m off at contact, at most
0.159 m; in 15 the plan in force was made at the bare robot radius. Whether
that fallback is cause or symptom is untested.

### Consequences

The explanation of the motion cost stands, but what it is worth without an
oracle is about half as much on dense clutter, before sensor noise. The next
test is cheap: withhold the zero-margin fallback from the estimating agent.

### Calibration

Prediction 18 failed. The headline -- bounded-inert on both conditions -- came
out COSTLY on dense clutter. The clauses registered for the case it failed put
the loss on dense clutter and said a wider margin would not recover it, both
right, but named the wrong channel: collisions, not timeouts. They are not
counted towards the prediction. A fallback written for the case the headline
fails does not earn the headline partial credit.

This is the calibration rule's own case. The prediction was arithmetic rather
than a measurement, and said so. The arithmetic held -- a median 0.030 m error
at contact, inside the 0.1 m it computed -- and the conclusion did not, because
the 0.18 m margin it was compared against was absent from the plan in force in
15 of 18 collisions. Phase 5q recorded an untested crossing that held, so that a
second could revise the rule if it held too. This one failed; the rule stands.

## Phase 5s — The encoder cost re-priced: it survives, and still stalls

Phase 3e's 0.16-0.24 cost of reading pixels was the last perception headline
measured under the 4:1 reward alone. Twelve runs, seed-paired with Phase 3e's
and identical to them except `collision_penalty` 5 -- the RGB arm pinned to
CUDA, as 3e's was, after a smoke test of the launch command showed `auto`
training it on CPU. Pre-registered in `26fee5e`, before any run started.

| rgbi − depthi | 4:1 | 1:1 | interaction | p |
|---|---|---|---|---|
| `narrow` success | -0.218 | -0.180 | +0.038 | 0.4697 |
| `narrow` collision | +0.105 | +0.027 | -0.078 | 0.3896 |
| `narrow` timeout | +0.113 | +0.153 | +0.040 | 0.7251 |
| `dense` success | -0.238 | -0.193 | +0.045 | 0.4697 |
| `nominal` success | -0.162 | -0.088 | +0.073 | 0.0195 |

At 1:1 every depth seed beats every RGB seed on `narrow` and `dense` (p = 0.002
on both). Registered decision: **SURVIVES**.

**The failure did not move.** Indifference collapses the depth arm's `narrow`
timeouts to 0.028; the RGB arm still times out on 0.182 of episodes,
and the encoder's cost there is timeouts (+0.153, p = 0.017, positive
on 6 of 6 seeds), not collisions (+0.027, p = 0.662).
Coverage's deficit moved into collisions under exactly this change (Phase 5l);
the encoder's did not.

`nominal` roughly halves, −0.162 to −0.088, interaction p = 0.0195 -- outside
the registered decision, which named `narrow` and `dense`, and not significant
against 0.05/3 for three conditions. Reported, not claimed.

### Consequences

The representation headline now holds under both rewards, as coverage's does.
But it is the one perception deficit whose failure the reward does not move: a
policy short of *information* crashes once stalling is priced like crashing,
and a policy with the information in a harder encoding keeps stalling. Why is
untested.

### Calibration

Prediction 19. The headline held: significant on `narrow` and `dense`, −0.180
inside the registered −0.13 to −0.30, success interactions of +0.038 and
+0.045 inside ±0.08 at p = 0.4697. The mechanism clause failed on both halves:
`narrow` timeouts +0.153 against a registered ±0.05, collisions +0.027 against
at least +0.15. Partial, by the standard applied to predictions 16 and 17.

It was registered LOW, as an analogy, and it split where the analogy was
weakest. Success invariance carried over from two perception re-pricings. The
mechanism was reasoned from timeouts collapsing for MLP arms at 1:1, and did not
carry over to a CNN arm, which nothing had measured.

## Phase 5t — The zero-margin fallback: a symptom, not the cause

Phase 5r's post hoc replay found the plan in force at the bare robot radius in 15
of 18 episodes the estimate lost. Tested by withholding that radius from movers
(`mover_margin_floor` 0.5), on the estimating agent and on the oracle as a
control. Pre-registered in `df8088e`, before the result file existed. Full
treatment in `dynamic_obstacles.md`.

Checks: cv_m2 and oracle_m2 reproduce Phase 5r on every episode. Identity: every
episode that never reaches the bare radius is bit-identical with the floor --
184, 169, 198 and 194 episodes across the four cells, none differing.
Registered decision: **SYMPTOM**.

- Estimate, fallback withheld against not, dense: +0.010, p = 0.5, 2 won
  and 0 lost, CI [+0.000, +0.025] -- bounded-INERT. Sparse: identical
  outcomes, though 16 episodes reach the fallback.
- Oracle control: bounded-INERT on both conditions (dense −0.005,
  CI [−0.015, +0.000]).
- The estimate reaches the fallback in 31 dense episodes against the oracle's 6,
  and 16 sparse against 2.
- With the fallback withheld the estimate still trails the oracle by 0.055 on
  dense clutter, p = 0.003.

### Consequences

The fallback is where a robot already too close to a mover ends up, not how it
got there. What is left is how it gets close: an estimate acted on for up to a
second between replans is the obvious candidate, and a replan triggered when an
observation contradicts the estimate would test it, with the oracle -- which is
never contradicted -- as a bit-identical control.

### Calibration

Prediction 20 failed. Its headline was CAUSE and the result is SYMPTOM. The
control clause and the identity held, but a control holding is what makes the
result readable, not a part of the forecast that came true, and it is not
counted. The rule used from here: a registered decision that fails is a failed
prediction, whatever its other clauses did; one that holds with a failed
secondary clause is partial.

It was registered LOW and leaned on a post hoc description of eighteen episodes.
The description has held up -- the estimate really does reach the fallback five
times as often -- and the inference drawn from it did not. Describing what
failures share is not measuring what causes them, which is the point of
registering a test before reading the description as an answer.

## Phase 5u — Replanning on a contradicted estimate: unresolved on dense clutter

Phase 5t: the fallback is where a robot already too close ends up. Tested here:
whether it gets close by acting on a stale estimate, replanning the moment an
observed mover is more than 0.05 m or 0.02 m from where the last plan's
estimate put it. Pre-registered in `b4ccdc2`, before the result file existed.
Full treatment in `dynamic_obstacles.md`.

Checks: the oracle with the trigger bit-identical to without on every
moving-world episode, the trigger never firing; cv_m2 and oracle_m2 reproduce
Phase 5r. Registered decision: **UNRESOLVED**.

- 0.02 m trigger against none, dense: +0.015, p = 0.549, 7 won and 4 lost,
  CI [−0.015, +0.050] -- neither MATTERS nor bounded-INERT. Collisions
  −0.020, timeouts +0.005; replans per episode 24.9 to 35.7.
- 0.05 m trigger, dense: +0.015, p = 0.453, at 6.9 triggered replans per
  episode against 29.2. No dose-response.
- Sparse: the 0.02 m trigger wins back all 4 episodes the estimate lost
  (p = 0.125, the floor for four) and its success matches the oracle's on
  every episode.
- With the trigger the estimate still trails the oracle by 0.050 on dense
  clutter, p = 0.002, 0 won and 10 lost.

### Consequences

Staleness may be the whole of the estimate's cost on sparse worlds and is at
most part of it on dense clutter, where 0.050 remains after a wider margin, a
withheld fallback and fresh estimates. The candidate registered first, in Phase
5r's prediction, and never tested is the far end of the window: a straight line
carried seven seconds out and held past it as a permanent obstacle. Capping how
far the estimate is extrapolated tests it, with frozen worlds as the identity.
Separately: at 200 episodes a dense-clutter null can only be bounded when the
effect is near zero, so a bounded answer there may need more episodes.

### Calibration

Prediction 21 failed: registered STALENESS, returned UNRESOLVED, and the dose
clause failed too -- equal gains at both thresholds. The control held and is
not counted. It was registered LOW, and the case written against it -- that the
approach is set by choices at the far end of the window -- is now the leading
candidate.

## Phase 5v — The pixel stall audit: the opening is seen, and not taken

Phase 5s left the encoding deficit as the one perception deficit whose failure
the 1:1 reward does not move. A training-free audit of the twelve Phase 5s
policies (`scripts/pixel_stall_audit.py`, `vision_nav.analysis.pixel_audit`).

**Not pre-registered.** The forecast in the script was written before any
result, but a smoke test printed output before it was committed (`32e1a27`
says so), and three descriptive fields were added after runs, each re-run and
checked to reproduce every earlier number exactly. Not counted in the
calibration record.

**Part 1, information.** Widest span of distances rendering to an identical
image column: walls 0.069 m, boxes 0.091 m, circles 0.114 m; medians
0.046, 0.048 and 0.060 m. The same below 1.2 m, where slice height
saturates and only shading carries distance. Information held, to about a
tenth of a metre.

**Part 2, anatomy**, 50 `narrow` worlds per policy:

| | depthi | rgbi |
|---|---|---|
| share of steps stalled | 0.076 | 0.350 |
| stall steps: goal route open and in view | 0.000 | 0.187 |
| stall steps: opening in view, goal route not | 0.964 | 0.812 |
| ... of which the goal is out of view | 1.000 | 0.856 |
| final-layer probe, balanced accuracy | 0.902 | 0.813 |

Stall share p = 0.022; probe p = 0.004 (exact permutation over seeds). The
registered rule returns BLIND DETOUR, and the label is wrong about what it
names: the depth policy's stalls look the same, and nearly all are a goal
outside the field of view. The RGB-specific stall is in front of an open route
to a visible goal, concentrated in seeds 3-5.

At those stalls the final-layer probe reads *blocked* 51% of the time
(2281 of 4466), against 15% on open-route steps while moving -- a
phantom obstacle, apparently. The same probe on the CNN's image features alone
reads blocked 24% of the time at those stalls and 35% while moving.
The final layer also takes the robot's own velocity, near zero at any stall.
The image features see the opening; the policy does not take it.

### Consequences

The encoder's cost is not lost information and, at the stalls that are
specific to it, not a failure to see the way on. Something downstream of the
features holds a stopped policy stopped. The velocity input is the obvious
candidate and can be tested without training: replay the stall states with it
set to a moving value.

### Calibration

Not counted. Recorded anyway: every clause of the forecast held -- information
within 0.15 m, more stalling, detour over half the stalls and over-represented,
goal_open under a quarter, a probe gap of at least 0.05 -- and the story it
told, a policy blind to the detour, is contradicted by the splits added
afterwards. Clauses that all hold can still be too coarse to tell the right
story from the wrong one.

## Phase 5w — Capping the estimate's reach: the far end is used, not a cost

Phase 5u left 0.050 of the estimate's dense-clutter cost after fresh estimates.
Tested here: the far end of the window, by carrying each straight line at most
2 s (primary) or 1 s past the latest observation. Pre-registered in `231c23b`,
before the result file existed. Full treatment in `dynamic_obstacles.md`.

Checks: capped and uncapped bit-identical on every frozen episode; the uncapped
arm reproduces Phase 5r. Registered decision: **UNRESOLVED**.

- 2 s cap against none, dense: +0.010, p = 0.727, 5 won and 3 lost,
  CI [−0.015, +0.040]. Dense episodes reaching the bare-radius fallback fall
  from 31 to 21.
- 2 s cap, sparse: −0.020, p = 0.289.
- 1 s cap, sparse: −0.085, p = 0.0005, 3 won and 20 lost, all of it
  collisions (+0.085). Dense: −0.035, p = 0.143.
- With the 2 s cap the estimate trails the oracle by 0.055 on dense
  clutter (p = 0.003) and 0.040 on sparse worlds (p = 0.008).

### Consequences

The far end of the window is information the planner uses, wrong in detail as it
is: take it away and the robot collides more. Four
planner-side explanations of the estimate's dense-clutter cost -- margin,
fallback, staleness, reach -- have now failed to account for it. The model has
not been touched: a straight line drawn along a sinusoid. A better estimate, not
a different use of this one, is the next test.

### Calibration

Prediction 23 failed: registered REACH, returned UNRESOLVED. Its secondary
clause held -- fewer episodes reach the fallback -- and, as in Phase 5t, that
changed nothing that mattered. The case written against it was half right:
withholding the fallback had indeed changed nothing, and believing movers stop
does cost collisions, at 1 s badly.

## Phase 5x — The velocity latch: it holds the stall, and releasing it crashes

Phase 5v found the RGB policy stopping in front of open routes its CNN features
read as open, with its final layer -- which also takes the robot's own velocity
-- reading them as blocked. Tested here with a replay of the twelve Phase 5s
policies over the same 50 `narrow` worlds, overwriting only the velocity input
(`scripts/stall_counterfactual.py`). Pre-registered in `da369f8`, before any
result existed; the smoke test printed only tracebacks.

**Open loop**, share of steps commanding at least 0.15 m/s forward:

| | real input | told moving (0.3 m/s) | told stopped |
|---|---|---|---|
| rgbi, open-route stall steps | 0.013 | 0.380 | -- |
| rgbi, other stall steps (opening in view) | 0.015 | 0.441 | -- |
| rgbi, open-route moving steps | 0.992 | -- | 0.984 |
| depthi, stall steps (opening in view) | 0.007 | 0.006 | -- |
| depthi, open-route moving steps | 0.999 | -- | 1.000 |

Exit shift +0.367, entry shift +0.008. Registered decision: **PARTIAL**.
The latch runs one way: velocity does not stop the RGB policy, but keeps a
stopped one stopped. The depth policy has none.

**Closed loop**, the velocity input set to 0.3 m/s whenever the robot has gone
nowhere for 3 s. RGB: timeouts −0.073 (p = 0.031, lower on all six seeds),
collisions +0.063 (p = 0.031, higher on all six), success +0.010
(p = 0.25). Depth: timeouts +0.010, collisions −0.010, success unchanged.

### Consequences

The latch is real, RGB-specific, and not what costs the episodes: released, the
stalls become crashes rather than successes, a trade the 1:1 reward prices at
nothing. Stopping was the policy's way of not crashing. Why it cannot go on is
the question left -- where the released episodes crash, and whether its features
encode the geometry around an opening rather than just the opening.

### Calibration

Prediction 22 failed: registered LATCH, returned PARTIAL. The exit clause held
(+0.367 against 0.30) and so did the closed-loop timeout clause (−0.073 against
0.05); the entry clause failed (+0.008 against 0.30). By the counting rule, a
failed registered decision is a failed prediction. The case written against it
-- that velocity shapes turning rather than going -- was wrong too: stopped and
told it is moving, the policy drives on 38% of open-route stall steps, not 1.3%.

## Phase 5y — Fitting the oscillation: the model was the whole of it

Phases 5r to 5w changed how the planner uses a constant-velocity estimate and
recovered none of its dense-clutter cost. This changes the model: each mover's
oscillation fitted to its observed track by three-unknown least squares, carried
forward in closed form. Pre-registered in `5091da1`, before the result file
existed, with the estimator's own error measured first and disclosed.

Checks: fitted and line agents bit-identical on every frozen episode; the line
arm reproduces Phase 5r. Registered decision: **MODEL**.

- Fitted against line, dense: +0.070, p = 0.0001, 14 won and 0 lost,
  CI [+0.035, +0.105]. Collisions 0.075 to 0.005.
- Fitted against line, sparse: +0.020, 4 won and 0 lost (p = 0.125, the floor
  for four).
- Fitted against **oracle**: +0.005 on dense, CI [+0.000, +0.015], bounded-INERT;
  on sparse, not one episode differs.
- Estimator error, no planner, median against the truth on `dynamic_dense`:
  the line 0.023 m at 1 s and 1.192 m at 7 s; the fit 0.0000 m at every
  horizon, p95 0.0001 m. (The pre-registration quoted the sparse figure,
  0.021 m at 1 s, for the same quantity.)
- Motion cost: 0.005 dense and 0.000 sparse, against the oracle's 0.010 and
  0.000 and the line's 0.075 and 0.020.

### Consequences

None of the 0.065 belonged to the planner. Margin, fallback, replan trigger and
capped reach were each a way of coping with a wrong estimate, and when the
estimate stopped being wrong the cost went with it. The report's motion-cost
claim no longer needs an oracle: a planner estimating from its own observations
does as well.

The caveat is the size of the claim. Observations here are noise-free and the
fitted model is the one the simulator integrates, so this is an upper bound --
what better prediction is worth, not what a real sensor would deliver. Sensor
noise on the same fit is the next test.

### Calibration

Prediction 24 held, on every clause: MATTERS on dense clutter (+0.070 against a
registered +0.03), bounded-inert against the oracle on both conditions, frozen
identity. The fourth of twenty-four to hold, and the fourth derived from a
measurement taken in the regime it was applied to -- here the estimator's own
error, measured before registering and disclosed in the commit, which made the
planner-level prediction close to arithmetic. Every category that has held in
this project is that one.

## Phase 5z — Noise on the observations: it costs more than the model bought

Phase 5y's fitted oscillation matched the oracle from exact observations. Here
each observed mover position carries a Gaussian error, 2 mm to 5 cm. Two stages:
the estimators' own error over the grid with no planner, then the planner at
levels chosen by a rule fixed before the sweep (a centimetre, plus the coarsest
level where the fit still won at two seconds -- which turned out to be 0.05).
Pre-registered in `f20f7e1`. Full treatment in `dynamic_obstacles.md`.

Checks: the oracle bit-identical with noise and without, and reproducing Phase
5r. Registered decision: **UNRESOLVED**.

- Fitted minus line at σ = 0.01, dense: +0.005, p = 1.0, 15 won and 14 lost,
  CI [−0.050, +0.055] -- neither MATTERS nor bounded-INERT.
- Fitted minus line at σ = 0.05, dense: +0.115, p = 0.0002, 30 won and 7 lost.
- Against their noise-free selves at σ = 0.01, dense: the line −0.095
  (p = 0.0013), the fit −0.160 (p < 0.0001).
- Fitted at σ = 0.01 against the oracle, dense: −0.155 (p < 0.0001).
- Stage 1, dense, 2 s median error: line 0.086 → 0.443 → 2.093 as σ goes
  0 → 0.01 → 0.05; fit 0.000 → 0.688 → 0.715. At 7 s the fit is bounded
  (0.93 m at σ = 0.05) where the line diverges (7.25 m).

### Consequences

The estimator chain reads: with exact observations the model was everything;
with a centimetre of error it is worth nothing measurable, and the error costs
more than the model ever bought. The fit's advantage returns only at coarse
noise, where its bounded orbit beats a diverging line.

Neither estimator smooths. Differencing raw observations twice to read curvature
divides the error by dt squared, which is what the sweep is measuring; a filter
that tracks position, velocity and frequency jointly is the repair, with 0.160
of dense-clutter success to win back.

### Calibration

Prediction 25 failed: registered FRAGILE, returned UNRESOLVED. Two clauses held
-- at a centimetre the fit is not better than the line, and both are far behind
the noise-free fit -- and two failed: the fit does *not* lose to the line at
every level from 5 mm up (it wins at 0.02 and 0.05), and at 0.01 it is 1.6 times
the line's error, not the tenfold registered.

The arithmetic behind it was right about the mechanism and wrong about the
consequence, which is Phase 5r's lesson repeated: a second difference does
amplify the error by dt squared, and the fit is duly swamped -- but a swamped
fit degrades into a bounded orbit, and a bounded wrong answer beats an unbounded
one. Nothing in the calculation said what the *failure mode* of each estimator
would look like.

## Phase 6a — Fitting without differencing: the filter closes the gap

Phase 5z left the estimator chain with a centimetre of observation noise costing
more than the motion model had ever bought, and diagnosed the cause: the fit
read curvature through a second difference. Here the same model is fitted to the
observations directly -- frequency searched, the rest by least squares over the
whole track. Pre-registered in `55bdca0`, before the result file existed, with
the estimator's error measured first and disclosed. Full treatment in
`dynamic_obstacles.md`.

Checks: frozen cells match the oracle's episode for episode; the noise-free arm
is bounded-INERT against the oracle. Registered decision: **FILTER**.

- Against the line at σ = 0.01, dense: +0.145, p < 0.0001, 32 won and
  3 lost. Sparse: +0.065, p = 0.004.
- Against the differencing fit at the same noise: +0.140 dense,
  +0.100 sparse.
- Against the oracle: −0.015 dense (p = 0.45), −0.020 sparse
  (p = 0.125) -- no longer distinguishable.
- Six seconds of history against three: −0.010 dense, p = 0.73.
- Estimator error at σ = 0.01, dense, no planner: 0.050 m at two seconds
  against the line's 0.295 and the differencing fit's 0.438.

### Consequences

The estimator chain closes: what it needed was not better information, nor a
better model class, but an estimator that does not amplify its own error. A
classical stack that observes mover positions to a centimetre, fits them
properly and plans in space-time is, on these worlds, indistinguishable from one
handed the exact future.

The remaining gift is larger than exactness was: the agent sees every mover at
every step, with no field of view and no occlusion. Taking that away is the next
test, and report Section 12 item 1 now says so.

### Calibration

Prediction 26 is partial. The headline held and then some -- registered at least
+0.05 on dense clutter against the line, measured +0.145, and the same against
the differencing fit -- as did the noise-free control and the frozen identity.
The clause that failed was the cautious one: that an estimate 0.05 m out at two
seconds would still trail the oracle measurably. It does not, at two hundred
episodes.

That is the fourth prediction in this project derived from measuring the
estimator before registering, and the fourth whose *magnitude* was right. What
it got wrong was again a claim about something it had not measured -- how the
planner converts a small estimate error into a lost episode.

A note on the numbers it quoted: the pre-registration cited 0.057 m at two
seconds from a ten-second warm-up, and the experiment's own measurement, with a
six-second one, gives 0.050 m. Neither was tuned; the warm-up differs.

## Phase 6b — Where a released policy crashes, and the gap it cannot measure

Phase 5x released the RGB policy's velocity latch and its timeouts became
collisions. Two measurements, no training, pre-registered in `557076c`.

Registered decision: **MIXED**.

- Added collisions (timed out untouched, collided once released): 19. Within a
  metre of the stall *and* three seconds of the release: 9. The registered rule
  needed over half, and returns MIXED.
- Post hoc, and labelled so: 13 of the 19 are within a metre of the stall at any
  delay; the median is 0.51 m and 3.1 s. The place was right in the
  prediction and the timing was not -- a released robot edges into something
  beside the opening over a few seconds rather than driving into it at once.
- Probe for the geometry *around* an opening: the CNN's image features decode
  the goal-ward gap's angular width at R² = −0.293, worse than
  predicting the mean, where the depth vector reaches 0.220
  (p = 0.0022 over six seeds, and every RGB seed below every depth
  seed).
- The same features carry the clearance half a metre ahead nearly as well as
  depth (0.205 against 0.264, p = 0.132).

### Consequences

The encoding keeps how far away the obstacle ahead is and loses how wide the way
past it is. That is the quantity a 0.22 m disc in a corridor needs, it is what
Phase 5v's stalls were in front of, and it is what a released policy fails at
half a metre from where it stopped. Report Section 12 item 2 is now the
encoder's resolution rather than its reward: the first convolution strides four
across a 64-pixel image, so a two-column gap cannot survive it.

### Calibration

Prediction 27 failed on its registered decision (AT THE OPENING, returned
MIXED) and held on its probe clause, which was the more specific one: at least
0.10 of R-squared behind on gap width, measured 0.513 behind. The conjunction
that failed -- a metre *and* three seconds -- was written without measuring how
long a released robot takes to reach anything, and the distance half of it was
right. A bound picked for a quantity nothing had measured is the same mistake
Phase 5r's arithmetic made, in a smaller way.

## Phase 6c — Seeing only what a sensor could see: the field of view is the cost

Phase 6a's estimator observed every mover at every step. Here it observes a mover
only when in range (6 m), in view, and not occluded by static geometry; unseen
movers coast, never-seen movers are absent from the planner's grid. Three arms at
Phase 6a's centimetre of noise. Pre-registered in `da22fce`, before the result
file existed. Full treatment in `dynamic_obstacles.md`.

Checks: the see-everything arm reproduces Phase 6a on every episode. Registered
decision: **UNRESOLVED**.

- 360° scanner against full sight, dense: −0.035, p = 0.092, 3 won and 10
  lost, CI [−0.070, +0.000]. Sparse −0.015.
- 90° camera against full sight, dense: −0.125, p < 0.0001, 1 won and
  26 lost; collisions 0.020 to 0.155. Sparse −0.085, p = 0.0002.
- Against the straight line with full sight (Phase 5z): scanner +0.110
  (p = 0.0001), camera +0.020 (p = 0.61).
- Against the oracle, dense: scanner −0.050 (p = 0.002), camera
  −0.140.

### Consequences

Occlusion by walls is cheap for a sensor that looks all round; a forward-facing
field of view is not, and costs nearly everything the estimator chain had won.
It is the report's coverage finding (§8.1) seen from the classical side: what a
moving world asks of a sensor is first that it look sideways.

Taking sight away leaves the largest privilege standing -- the static map and
the exact pose, both given to the classical stack from the start and never to
the policy. Even with a camera's view of movers it keeps 0.835 on dense clutter
against the best learned policy's 0.710.

### Calibration

Prediction 28 is partial. Held: the scanner's cost landed in its registered band
(−0.035 against −0.02 to −0.10) and was not a bounded null; the camera cost at
least 0.10 and was HARMS (−0.125). Failed: the camera arm was predicted to stay
far ahead of the straight line with full sight, and is not (+0.020, p = 0.61) --
the clause assumed the estimator would keep its advantage on whatever it could
see, without measuring how much a 90-degree view leaves it to see. The scanner's
cost is not significant, so "seeing less costs something" is established for the
camera only.

## Phase 6d — The map test, as registered: it measured two bugs of mine

The report's Result 1 was measured with the planner handed a perfect static map.
Here it builds its own from its sensor as it drives, pose still exact
(`src/vision_nav/mapping/occupancy.py`, `src/vision_nav/agents/mapped.py`). Pre-registered in `915c76a`.

The full-map arm reproduces the published classical row on all six conditions.
Registered decision: **COSTLY** -- and it is not a finding about mapping.

| success, 100 worlds | full map | own map, 32-beam scanner | own map, 90° camera | PPO |
|---|---|---|---|---|
| sparse | 1.000 | 1.000 | 1.000 | 0.980 |
| nominal | 1.000 | 0.990 | 0.960 | 0.960 |
| large | 1.000 | 0.980 | 0.990 | 0.970 |
| dense | 0.890 | 0.610 | 0.660 | 0.640 |
| narrow | 0.850 | 0.540 | 0.590 | 0.600 |
| noisy_lidar | 1.000 | 0.540 | 0.520 | 0.960 |

A drop that large would overturn Result 1 in clutter, so before reading it a
post hoc diagnostic (`scripts/mapping_diagnostic.py`) asked what the robot hit.
On `dense` and `narrow`, all 79 collisions across both sensors were into
obstacles the map had held for **at least a second** before contact -- a median
of 63 to 121 scans, six to twelve seconds -- and none into one mapped in the last
half second. The robot drove into what it already knew was there. That is a
fault in the planner I built, not the limit of the sensor.

Two faults, both mine and both departures from standard mapping practice:

1. **Surface extent.** The map took each obstacle surface to sit at the centre of
   its occupied cell. It can sit anywhere in it, up to half a cell diagonal
   (about 0.07 m) closer, so inflation, the smoother's line-of-sight check and
   the controller's slow-down all overestimated clearance, and paths grazed
   what the full-map planner, reading exact geometry, keeps clear of.
2. **Evidence under noise.** Occupied cells were sticky -- correct for a
   noise-free sensor, and wrong for `noisy_lidar`, where every return landing
   short marked free space occupied for good and phantom obstacles accumulated.
   A real mapper counts evidence for and against.

Neither is a finding about what mapping costs, so the report's Result 1 stands
as written, and this result is not quoted there. The repaired planner is re-run
under its own pre-registration; the repairs are made and checked on the `val`
seed band, which the experiment never scores, so that pre-registration can be
clean.

### Calibration

Prediction 29 is recorded and not scored: its registered decision (COSTLY) was
reached, but by a planner that could not do what the prediction was about.
Counting it either way would score the bug.

## Phase 6e — The map, repaired and re-run: it was worth the clutter margin

Phase 6d's run measured five faults in the mapping stack. Repaired -- conservative
surface placement, log-odds evidence, never driving a plan the map rules out,
footprint clearing, goal relaxation, rotate-in-place recovery, and replanning
rate limited to 1 Hz for anything not close ahead -- and re-registered in
`34cf3bc` with a prediction derived from checks on the `val` band, which this
experiment does not score. Full treatment in report §9.2.

The full-map arm reproduces the published classical row on all six conditions.
Registered decision: **COSTLY**.

| success, 100 worlds | given the map | own map, scanner | own map, camera | PPO |
|---|---|---|---|---|
| sparse | 1.000 | 1.000 | 1.000 | 0.980 |
| large | 1.000 | 0.990 | 0.990 | 0.970 |
| nominal | 1.000 | 0.970 | 0.980 | 0.960 |
| noisy_lidar | 1.000 | 0.940 | 0.930 | 0.960 |
| dense | 0.890 | 0.650 | 0.690 | 0.640 |
| narrow | 0.850 | 0.590 | 0.610 | 0.600 |

- Dense: −0.240, p < 0.0001, CI [−0.330, −0.150]. Narrow: −0.260,
  p < 0.0001, CI [−0.350, −0.180]. Open conditions: −0.030 at worst.
- The cost is timeouts (0.330 dense, 0.390 narrow), not collisions
  (0.020 on both). About 75 replans an episode, half the world still
  unmapped at the end.
- `noisy_lidar`, which a perfect map made a no-op for this stack, now costs
  0.060 (p = 0.031).
- The camera's 64 columns over 90 degrees beat the scanner's 32 over 360 in
  clutter, on both conditions -- forward resolution over coverage, when what is
  being mapped is static and the robot is driving into it.

### Consequences

Result 1's margin in clutter was mostly the map. Given the same scan the policy
reads, the planner's 0.25 lead on `dense` and `narrow` becomes 0.650 against
0.640 and 0.590 against 0.600 -- a tie at a hundred episodes either way, and the
comparison is descriptive: the PPO column is one run's aggregate with no
per-episode outcomes to pair against. In open worlds the planner keeps its lead.

The report's headline now carries that qualifier, in the abstract and in the
Result 1 table, which gained the column rather than the finding being buried in
a later section. What is measured is this mapping stack, which §11 says plainly;
the pose is still exact, and Nav2 with SLAM is the production comparison.

### Calibration

Prediction 30 is partial. The registered decision (COSTLY) was reached, both
magnitude bands held (−0.240 in [−0.12, −0.25]; −0.260 in [−0.15, −0.28]), the
collision ceiling held (0.020 against 0.05), the open conditions held (−0.030
against 0.03) and `noisy_lidar` held (0.940 against 0.90). The clause that failed
was "still beats the privileged PPO policy on dense and narrow": on narrow it
lands at 0.590 against 0.600.

The prediction came from val-band measurements of the same repaired stack, which
is the category that has held before -- and did hold here, to within 0.03 on
every cell. What it got wrong was the comparison it had not measured: where the
policy sits.

## Phase 6f — Take away the pose: the last privilege

§9.2 left the planner its exact pose and called it the last privilege it held.
This takes it. The pose comes from wheel odometry carrying both errors a real
base has — the random walk of the standard motion model, and a scale and
heading bias drawn once per robot — and, in the arm that matters, from that
estimate corrected by matching each scan against the map the robot is building.
Registered in `90724be` with a prediction derived from val-band measurement,
before any test world was scored. Full treatment in report §9.3.

Registered decisions: **COSTLY** (the pose was worth a great deal) and **PAYS**
(scan matching gets most of it back).

| success, 100 worlds | given the map | own map | odometry | odometry + matching | PPO |
|---|---|---|---|---|---|
| sparse | 1.000 | 1.000 | 0.670 | 0.570 | 0.980 |
| large | 1.000 | 0.990 | 0.300 | 0.530 | 0.970 |
| nominal | 1.000 | 0.970 | 0.700 | 0.850 | 0.960 |
| noisy_lidar | 1.000 | 0.940 | 0.640 | 0.780 | 0.960 |
| dense | 0.890 | 0.650 | 0.410 | 0.640 | 0.640 |
| narrow | 0.850 | 0.590 | 0.420 | 0.630 | 0.600 |

- Dead reckoning alone is HARMS on all six, from −0.170 on `narrow` to −0.690
  on `large`, all p < 0.001, and the size tracks the distance driven.
- Matching is MATTERS on five of six, and in clutter the pose then costs
  nothing measurable: −0.010 on `dense`, +0.040 on `narrow`. Median pose error
  0.084 m and 0.083 m, under a degree of heading.
- **The exact inverse of Phase 6e.** The map cost −0.240/−0.260 in clutter and
  nothing in the open; the pose costs nothing in clutter and −0.460/−0.430 in
  the open. Structure is what a map is needed for *and* what a pose is
  recovered from.
- `sparse` is the one place matching hurts (0.570 against 0.670). Its own score
  stays high while the pose is wrong: the map is built at the estimate and the
  estimate matched against the map, so both drift together. No back end.
- The 90° camera's clutter advantage from 6e reverses on `narrow` (0.530
  against the scanner's 0.630, p = 0.0063) and holds on `large` (+0.140).

### Controls

Three, all passing: the full-map arm reproduces the published classical row on
all six conditions; the own-map arm reproduces Phase 6e's per-episode outcomes
on all six; and the same localised agent with the odometry noise off is
bit-identical to it.

### Calibration

Prediction 30 is partial. The two registered decisions were both reached, and
every clause held — the HARMS bands on `narrow` and `dense`, `large` being the
worst of the six, the MATTERS bands for matching, the near-zero residual in
clutter, the pose-error bands, the collision ceiling and all three controls —
except one. It predicted matching would fall below dead reckoning on `sparse`
"by between 0.15 and 0.50"; it fell below by 0.100, and inconclusively
(p = 0.184). The direction was right, from a mechanism measured on val; the
magnitude was not.

## Phase 6g — Ask a production stack the same question

§9.2 and §9.3 measured what the map and the pose were worth to this project's
stack and both ended on the same caution. This asks Nav2 with slam_toolbox —
no map, no pose, §9.3's odometry imported — at 360 beams and at the
hand-written stack's 32. The statistic is the difference in costs, world by
world, pooled: what Nav2 loses without its privileges minus what the
hand-written stack loses. Registered in `cb26af9`. Full treatment in report
§9.4.

Registered decisions: **UNRESOLVED** at 32 beams (the primary, like for like),
**IMPLEMENTATION** at 360.

| success, 100 worlds | hand-written, with / without | Nav2, with | Nav2 + SLAM, 360 | Nav2 + SLAM, 32 |
|---|---|---|---|---|
| sparse | 1.000 / 0.570 | 0.990 / 0.990 | 0.960 | 0.690 |
| large | 1.000 / 0.530 | 0.990 / 0.990 | 0.960 | 0.400 |
| nominal | 1.000 / 0.850 | 0.980 / 0.970 | 0.970 | 0.800 |
| noisy_lidar | 1.000 / 0.780 | 0.970 / 0.980 | 1.000 | 0.750 |
| dense | 0.890 / 0.640 | 0.940 / 0.910 | 0.820 | 0.620 |
| narrow | 0.850 / 0.630 | 0.930 / 0.910 | 0.810 | 0.450 |

- Pooled costs: hand-written 0.290; Nav2 at 360 beams 0.047 and 0.038 against
  the two full-privilege passes; Nav2 at 32 beams 0.348 and 0.340.
- 360 beams: difference +0.243 [+0.203, +0.285] and +0.252 [+0.212, +0.292].
  In clutter Nav2 pays 0.09 to 0.12, in timeouts, with pose error under 8 cm:
  the map, not the pose.
- 32 beams: −0.058 [−0.108, −0.008] and −0.050 [−0.100, +0.000]. Nav2 pays at
  least as much; better on `sparse`, much worse in tight corridors (0.450
  against 0.630, p = 0.0039).
- What the privileges stood in for was mostly the sensor. The hand-written
  stack at 360 beams — the missing cell — was not run.

### What it took

Seven faults found on the val band, all this project's: slam_toolbox launched
as a plain node when it is a lifecycle node; an unthrottled startup loop; Nav2's
commands queued behind TF traffic in a shared executor; a global costmap too
small for unseen goals; startup races between parallel runs; a cleanup script
defeated by Linux's fifteen-character process names; and — the one that
mattered — this project's own "scaled-down" slam_toolbox thresholds, which made
SLAM drag the pose backwards with perfect odometry. `slam_probe.py` found that
one in four runs.

### Calibration

Prediction 31 is partial. Held: IMPLEMENTATION at 360 beams inside its
+0.15 to +0.35 band; the 32-beam point estimates inside −0.07 to +0.07; at
least 0.93 on the four open conditions at 360; 32 below 360 everywhere and
lowest on `large`; open-world timeouts at 32 of at least 0.15; pose error under
0.20 m at 360 and larger at 32 everywhere; the starvation gate and the
SLAM-readiness bound. Failed: the primary decision (UNRESOLVED, not PROBLEM —
the intervals reached −0.108 and −0.100 against a ±0.10 band); `dense` at 360
fell 0.09 below the second full-privilege pass, not 0.10; and the mechanism.
Val had shown clutter losses at 360 as collisions, 0.15 and 0.20; on test they
were 0.040 and 0.010, and the loss was timeouts. Three or four collisions in
twenty worlds were never a mechanism.

## Phase 6h — The missing cell: this stack at 360 beams

§9.4 concluded from Nav2 alone that what the privileges stood in for was
mostly the sensor. This runs the inferred cell — the hand-written stack,
without map or pose, at 360 beams, every parameter unchanged — and corrects
that conclusion. Registered in `8a3ea6a`. Full treatment in report §9.5.

Registered decision: **IMPLEMENTATION** — the difference in costs against Nav2
at 360 beams is +0.112 [+0.077, +0.147] and +0.120 [+0.085, +0.155].

| success, 100 worlds | given the map | own map, 32 | + own pose, 32 | own map, 360 | + own pose, 360 | Nav2 + SLAM, 360 |
|---|---|---|---|---|---|---|
| sparse | 1.000 | 1.000 | 0.570 | 1.000 | 0.970 | 0.960 |
| large | 1.000 | 0.990 | 0.530 | 0.990 | 0.900 | 0.960 |
| nominal | 1.000 | 0.970 | 0.850 | 0.970 | 0.960 | 0.970 |
| noisy_lidar | 1.000 | 0.940 | 0.780 | 0.830 | 0.750 | 1.000 |
| dense | 0.890 | 0.650 | 0.640 | 0.640 | 0.620 | 0.820 |
| narrow | 0.850 | 0.590 | 0.630 | 0.590 | 0.590 | 0.810 |

- The dense scanner halves this stack's cost, 0.290 to 0.158 (+0.132
  [+0.097, +0.167]), and all of it is the pose: 0.190 to 0.038.
- The map's cost does not fall (0.100 to 0.120), and under noise the map gets
  worse with more beams (0.940 to 0.830 on `noisy_lidar`).
- Against Nav2 at 360 the residual is mapping: this stack loses 0.27 to 0.28
  more under noise and 0.14 to 0.18 more in clutter, 85% of the residual.
  In open, noise-free worlds the two lose nearly the same.
- §9.4 corrected: the pose stood in for the sensor; the map stood in for the
  implementation.

### Controls

full_map, mapped32 and matched32 reproduce Phase 6f per episode on all six.

### Calibration

Prediction 32 is partial. Every quantitative clause held: the difference in
costs inside its +0.05 to +0.15 band against both passes, the decision one of
the two named, noisy_lidar the largest single contribution, the cost at 360,
the scanner's effect, the pose and map costs, mapped360 below mapped32 on
noisy_lidar, sparse at least 0.90, both pose-error clauses and the three
controls. What failed was the headline's attribution: "what is left is mostly
this stack's map under sensor noise." It is the map, but noise carries 38 to
42% of the residual and clutter 43 to 47%.

## Phase 6i — A mapper for dense scans

§9.5 left this stack's mapper as what it pays at 360 beams.
`dense_map_diagnostic.py` measured the map against the world it was built from
on the val band: under noise, ten times the returns triple the phantom cells
(104 to 370), block 46% of the arena where the truth is 38%, and lose 28% of
the free floor; in clutter there are no phantoms at all and the map blocks less
than the truth. So the noise regression is the mapper's and the clutter gap is
not. The repair weighs a cell's returns against the returns a surface there
would have produced, and is inert for sparse scans. Registered in `6a222bf`.
Full treatment in report §9.6.

Registered decision: **UNRESOLVED** — the own-map arm on `noisy_lidar` gains
+0.080 at p = 0.0574, a whisker short of the MATTERS the prediction named.

| success, 100 worlds, 360 beams | own map | repaired | own map and pose | repaired |
|---|---|---|---|---|
| sparse | 1.000 | 1.000 | 0.970 | 0.990 |
| large | 0.990 | 0.990 | 0.900 | 0.890 |
| nominal | 0.970 | 0.960 | 0.960 | 0.960 |
| noisy_lidar | 0.830 | 0.910 | 0.750 | 0.890 |
| dense | 0.640 | 0.620 | 0.620 | 0.590 |
| narrow | 0.590 | 0.590 | 0.590 | 0.600 |

- With its own pose too — how the stack actually runs — `noisy_lidar` goes
  0.750 to 0.890: +0.140, p = 0.0005, MATTERS.
- Pooled +0.008 [−0.007, +0.023] own map, +0.022 [+0.002, +0.042] own pose.
- The difference in costs against Nav2 at 360 beams falls from +0.112 and
  +0.120 to +0.090 and +0.098: IMPLEMENTATION becomes UNRESOLVED.
- The mechanism is not the designed one, as the registration predicted: the
  phantoms barely move (370 to 309 on val); what falls is replanning, 132 to 91.

### Controls

Both uncorroborated arms reproduce Phase 6h per episode on all six conditions.

### Calibration

Prediction 33 is partial. Ten clauses held: both noisy gains inside their
bands, the other five conditions inside 0.06, both pooled bands, the shrink in
the difference against Nav2 (0.022 twice, inside 0.01 to 0.09), the collision
ceiling, and both controls. What failed was the registered decision: REPAIRED
needed MATTERS on the own-map arm and it came to p = 0.0574. The endpoint that
isolated the map best was the one with least power.

## Hardware notes

Development target is a laptop RTX 5070 Ti (12 GB VRAM), which is **below**
Isaac Sim's documented 16 GB minimum. Per the roadmap's Tier-1 guidance, the
whole pipeline is therefore built and debugged on this lightweight NumPy
simulator, with Isaac Lab / Habitat reserved for the final comparison runs on
rented GPU time. The task definition, metrics, splits and evaluation harness
all carry over unchanged — only the simulator behind `ProceduralNavEnv` swaps.
