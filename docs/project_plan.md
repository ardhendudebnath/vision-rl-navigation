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
`training/run_spec.py` exists, and why the benchmark now reports the sensor
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

## Hardware notes

Development target is a laptop RTX 5070 Ti (12 GB VRAM), which is **below**
Isaac Sim's documented 16 GB minimum. Per the roadmap's Tier-1 guidance, the
whole pipeline is therefore built and debugged on this lightweight NumPy
simulator, with Isaac Lab / Habitat reserved for the final comparison runs on
rented GPU time. The task definition, metrics, splits and evaluation harness
all carry over unchanged — only the simulator behind `ProceduralNavEnv` swaps.
