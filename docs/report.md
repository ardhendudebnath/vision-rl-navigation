# Where Learned Navigation Loses to Classical Planning, and Why

**A controlled comparison of reinforcement-learning and search-based navigation
under distribution shift**

Ardhendu Debnath — [vision-rl-navigation](https://github.com/ardhendudebnath/vision-rl-navigation)

---

## Abstract

We compare a PPO navigation policy against a classical A\* + pure-pursuit stack
on a procedurally generated point-goal task, under matched conditions and
across five environment distributions. The classical planner wins on every
condition, by margins that grow with clutter (1.000 vs 0.960 success in the
nominal setting; 0.850 vs 0.682 in cluttered "narrow" worlds). We then
systematically eliminate the standard explanations for a losing RL result:
insufficient data, wrong training distribution, insufficient compute, and
reward mis-specification are each tested and rejected. The surviving
explanation is sensor resolution, which we establish in three stages: a
single-seed experiment that appeared to confirm it, a four-seed replication
that **refuted** that confirmation, and a training-free measurement of the
sensor itself that quantified the mechanism, explained why the replication
failed, and predicted the effect size of a subsequent powered experiment that
did confirm it (+0.085 success, p = 0.035, pre-registered). Finally we show
that for this task **field of view dominates angular resolution**: a depth
camera with 4× finer angular sampling but a 90° field of view loses to a 360°
lidar on every condition (−0.097 success on the primary endpoint, p = 0.019).
Sweeping field of view at a fixed sample count turns that into a monotone
curve (Spearman ρ = +0.508, p = 0.013 over 24 seeds), with the audit's
out-of-sample predictions for the two interior levels landing within 0.021 and
0.001 of the measured means. Decoupling the two factors then shows the
relationship is not a trade-off at all: **doubling the sample count at fixed
field of view changes nothing** (+0.002 and −0.003, at 90° and 360°
respectively, both inside a pre-registered ±0.01 bound), while **quadrupling
coverage at identical angular resolution produces the entire effect** (+0.095,
p = 0.024). Angular resolution is second-order over this range; coverage is
causal.

Finally, holding the information constant and changing only the
representation — the same geometry rendered as pixels for a CNN rather than
read as a vector by an MLP — costs 0.16–0.24 success across every condition,
with every depth seed beating every RGB seed. On this task the pixels, not the
geometry, are the hard part.

Adding obstacles that move and are absent from the map — the one place a
reactive policy has a structural reason to win — shrinks the classical
planner's advantage roughly eightfold (−0.168 to −0.022) and leaves it the
only condition in the study where that advantage is not statistically
established (p = 0.062). It does not reverse it.

Training on movers and giving the policy stacked frames — the one setup
where it holds information the planner structurally lacks — changes
neither outcome (frame stacking: −0.008, p = 0.784). What emerges instead
is parity rather than victory: where the map is wrong the learned policy
is statistically indistinguishable from the planner (p = 0.625), while
remaining clearly worse where the map is right (−0.168, p = 0.031).

The most transferable contribution is methodological. A correctly computed
significance test over episodes produced a confident, reproducible, and wrong
conclusion, because it measured the wrong source of variance. We document that
failure in full, alongside the training-free measurement that diagnosed it and
then twice forecast the results of experiments not yet run.

---

## 1. Motivation

Learning-based navigation is usually evaluated against weak baselines, on the
distribution it was trained on, with a single training seed. Each of those
choices flatters the learned method. This project asks a deliberately
unflattering question:

> Where does a learned navigation policy actually beat a strong classical
> planner, where does it lose, and how does each degrade when the world stops
> looking like the training set?

Every design decision follows from wanting that comparison to be trustworthy
rather than favourable. The classical baseline is given privileges the learned
policies never get — the full obstacle map and exact pose — because a baseline
that loses through handicap proves nothing.

## 2. Related work

**Evaluation.** We report Success weighted by Path Length (SPL) following
Anderson et al., *On Evaluation of Embodied Navigation Agents* (2018), so
numbers are comparable to the embodied-navigation literature rather than being
project-specific scores.

**Classical stack.** The baseline mirrors the structure of ROS 2's Nav2: a
global planner over an inflated costmap, followed by a local controller
tracking that plan. This is deliberate — it means Nav2 itself can later replace
the baseline without changing the comparison protocol.

**Learning.** PPO (Schulman et al., 2017) via Stable-Baselines3. Domain
randomisation follows the standard recipe of widening the training distribution
to cover deployment conditions (Tobin et al., 2017).

**Seed variance.** Henderson et al., *Deep Reinforcement Learning That Matters*
(2018), documented that RL results vary substantially across training seeds and
that single-seed comparisons routinely mislead. Section 6 of this report is a
direct, unintentional replication of that finding.

## 3. Method

### 3.1 Task

A differential-drive robot must reach a goal in a procedurally generated arena
of circular and box obstacles. Worlds are generated from an integer seed and
are **guaranteed solvable**: start and goal are collision-free, at least 5 m
apart, and verified connected before the episode begins. Unsolvable layouts are
rejected and resampled.

That guarantee is load-bearing rather than cosmetic. If some episodes were
impossible, a failure would be ambiguous between "bad policy" and "bad task",
and both success rate and SPL would stop meaning anything.

- **Robot**: unicycle kinematics with acceleration limits. Actions are
  normalised `(v, ω)` — deliberately the same interface as
  `geometry_msgs/Twist`, so a real ROS 2 base is a transport change rather than
  a policy rewrite.
- **Observation** (privileged mode, 37-d): 32 normalised lidar ranges, goal
  distance, goal bearing as `(cos, sin)`, and current linear/angular velocity.
- **Episode limit**: 500 steps at 10 Hz.

### 3.2 Splits and shifts

Disjoint seed bands: `train` 0–999, `val` 10000–10099, `test` 20000–20199,
`test_ood` 30000–30199. Named distribution shifts (`dense`, `sparse`, `large`,
`narrow`) change obstacle density, size, or arena scale relative to training.

### 3.3 Metrics

Success rate, SPL, collision rate, timeout rate, steps-to-goal, path
efficiency. Two details materially affected the results:

**`l*` is string-pulled before use.** A raw 8-connected A\* path overestimates
the true geodesic distance. An overestimated `l*` makes `l*/max(p, l*)` clamp to
1.0 for any competent policy, and SPL silently stops discriminating. Before
this fix the baseline reported path efficiency of exactly 1.000 — saturated and
uninformative; after it, 0.993.

**Mean steps-to-goal averages successes only.** Including failures would let a
policy look fast by crashing early.

### 3.4 Reward

Progress reward is shaped on an A\* **geodesic** distance field rather than
Euclidean distance. Euclidean shaping creates a local optimum behind every
obstacle: the agent is paid to press into a wall that happens to lie between it
and the goal. The distance field is training-time privileged information only
and never enters the observation, so the policy remains honestly sensor-only at
evaluation.

### 3.5 Protocol

All actors are evaluated on **identical worlds in identical order**, asserted at
evaluation time rather than assumed. Model selection uses **validation SPL**,
not training return: return is shaped and not comparable across configurations,
so selecting on it risks picking a checkpoint that learned to farm the shaping
term.

## 4. Result 1: the classical planner wins everywhere

Success / SPL, 100 held-out worlds per condition:

| Condition | Classical | PPO (privileged) |
|---|---|---|
| nominal | **1.000** / 0.985 | 0.960 / 0.910 |
| sparse | **1.000** / 1.000 | 0.980 / 0.963 |
| large | **1.000** / 0.990 | 0.970 / 0.940 |
| dense | **0.890** / 0.841 | 0.640 / 0.593 |
| narrow | **0.850** / 0.795 | 0.600 / 0.556 |

The learned policy is competitive in open worlds and collapses under clutter.

The project's original hypothesis was the opposite. We had measured the
baseline's weakness precisely: under shift, **A\* never once fails to find a
route** — 0 planning failures across `dense` and `narrow`, with 9/11 and 12/15
of failures being the *controller* losing the plan while cornering. The
baseline's weakness is tracking under clutter, not planning under clutter, and
a reactive policy not committed to a precomputed path should in principle close
exactly that gap.

It does not. It degrades roughly three times as much.

**Governing caveat.** Both learned policies train on a distribution; the
classical planner has none. On shifted conditions this compares an
in-distribution planner against an out-of-distribution policy. That asymmetry
is inherent to comparing learned and non-learned systems and is most of the
explanation for Result 1.

## 5. Results 2–4: eliminating the standard explanations

### 5.1 It is not the training distribution

A second policy was trained with per-episode domain randomisation over arena
size, obstacle count, obstacle size and start-goal separation, with ranges
chosen to **contain** every evaluation shift. Success barely moved: 0.640 →
0.660 on `dense`, 0.600 → 0.630 on `narrow`, both inside the noise at n = 100.

### 5.2 It is not compute

The randomised policy was extended to 4.0M steps — 2.7× the nominal policy's
budget. Validation plateaus by ~1.75M and then drifts slightly down. Success
rose by +0.04 on `narrow` over 2.7× the compute.

*A correction worth recording:* an earlier version of this work discounted
Section 5.1 on the grounds that the policy was "still improving at 1.5M". That
rested on two points of a noisy 50-episode validation curve — noise read as a
trend. The compute sweep tested and rejected it.

### 5.3 The failure mode is caution, and it is rational

What *did* change across randomisation and compute was the **composition** of
failures. On `narrow`, 100 episodes:

| Policy | Successes | Collisions | Timeouts | Timeout progress |
|---|---|---|---|---|
| nominal, 1.5M | 60 | 27 | 13 | 4.4 m of 10.6 m, 0.09 m/s |
| DR, 1.5M | 63 | 11 | 26 | 7.5 m of 11.7 m, 0.15 m/s |
| DR, 4.0M | 64 | **2** | **34** | 7.1 m of 11.8 m, 0.14 m/s |

Collisions fall to 2 in 100: the policy has all but learned not to crash. But
of the 25 episodes that left the collision bucket, **21 became timeouts and
only 4 became successes**. The stalled episodes crawl at 0.14 m/s against a
0.6 m/s cap. The policy converges on caution: it learns not to crash, not how
to get through.

Measuring episode returns explains why. On `narrow` with the 4.0M policy:

| Outcome | Mean return |
|---|---|
| success | +41.20 |
| timeout | −2.06 |
| collision | −24.91 |

A collision costs a flat −20; timing out for all 500 steps costs −5. **Crashing
is four times worse than stalling forever.** The policy is not malfunctioning —
it found the optimum of the reward it was given.

### 5.4 It is not the reward balance either

Three arms each isolated one caution term. On `narrow`:

| Policy | Success | Collisions | Timeouts |
|---|---|---|---|
| classical | **0.850** | 0.120 | 0.030 |
| DR baseline | 0.630 | 0.110 | 0.260 |
| `proximity_penalty` 0.15→0 | 0.660 | 0.100 | 0.240 |
| `step_penalty` 0.01→0.05 | 0.660 | 0.280 | 0.060 |
| `collision_penalty` 20→5 | 0.560 | **0.440** | **0.000** |

As caution falls, timeouts convert into collisions almost one-for-one while
success stays pinned in a 0.56–0.66 band. **The caution terms control which
failure occurs, not how many.** Paired tests confirmed no arm significantly
improves success under clutter.

## 6. Result 5: a false positive, and how it was caught

This section is the methodological core of the report.

### 6.1 The apparent finding

The remaining hypothesis was sensor resolution. A planar lidar samples at fixed
*angular* spacing, so the linear gap between adjacent rays grows with range:

| Beams | Spacing | Resolves a 0.44 m robot-width gap out to |
|---|---|---|
| 16 | 22.50° | 1.13 m |
| 32 | 11.25° | 2.24 m |
| 64 | 5.62° | 4.48 m |
| 128 | 2.81° | 8.96 m |

At the 32 beams used throughout, the robot cannot reliably resolve a gap it
would fit through beyond 2.24 m — roughly one body length of lookahead.

Retraining at 64 beams gave `narrow` success +0.070, paired over 100 identical
episodes, 95% CI [+0.006, +0.134] — the first significant success improvement
under clutter in the project. Collisions appeared to fall from 0.110 to 0.030.

### 6.2 The refutation

Re-running both arms across **four training seeds each** dissolved it:

| Arm | Per-seed `narrow` success | Mean ± sd |
|---|---|---|
| 32 beams | 0.63, 0.68, 0.66, 0.52 | 0.623 ± 0.071 |
| 64 beams | 0.70, 0.58, 0.64, 0.74 | 0.665 ± 0.070 |

**+0.042, p = 0.457.** The seed-to-seed spread (±0.07) is larger than the
effect. The collision result was pure seed luck: across seeds the arms are
indistinguishable (0.077 ± 0.025 vs 0.080 ± 0.048). Seed 0 of the 32-beam arm
happened to be its *worst* for collisions and seed 0 of the 64-beam arm its
*best*.

### 6.3 Why a correct test gave a wrong answer

The episode-level confidence interval in §6.1 was correctly computed. Pairing
over episodes controls world difficulty and correctly answers *"do these two
policies differ on these worlds?"* — to which the answer was genuinely yes.

But the question that matters is *"does 64 beams beat 32 beams?"*, and that
requires treating the **training seed** as the unit of analysis. The episode
test was silent on the dominant source of variance. It was not a
miscalculation; it was a correct answer to the wrong question, which is
considerably harder to notice.

We therefore switched to an **exact permutation test** with the seed as the
unit. At four seeds per arm a *t*-test rests entirely on a normality assumption
four points cannot support, whereas enumerating all C(8,4) = 70 relabellings is
assumption-free. The cost is a resolution floor: the smallest two-sided *p*
obtainable from 4-vs-4 is 2/70 = 0.029, so "p < 0.05" there means "one of the
two most extreme of seventy splits" and nothing finer.

## 7. Result 6: measuring the mechanism directly

Rather than continue inferring perception limits from success rate, we measured
the sensor itself. Across 480 on-route poses in 60 `narrow` worlds: sweep 2048
ground-truth rays to find every *traversable direction* (a heading the robot's
disc can translate 3 m along without collision, checked against world geometry
rather than any sensor), group them into maximal gaps, and ask whether an
N-beam scan places at least one sufficiently long beam inside each gap. A gap
containing no beam is invisible to any policy, however good.

No policy, no training, no seeds.

| Beams | Gaps detected | 0–5° gaps | 5–10° | 10–20° | >20° |
|---|---|---|---|---|---|
| 16 | 0.778 | 0.129 | 0.451 | 0.635 | ~1.0 |
| 32 | **0.885** | 0.258 | 0.738 | 0.980 | 1.0 |
| 64 | **0.949** | 0.553 | 1.000 | 1.000 | 1.0 |
| 128 | 0.974 | 0.773 | 1.000 | 1.000 | 1.0 |

The deficit is real and sits exactly where the geometry predicts: a 32-beam
scan misses 11.5% of traversable gaps and 74% of gaps narrower than 5°.
Everything wider than 20° is seen by every sensor tested.

**This reconciles §6.1 and §6.2.** Going 32 → 64 beams recovers only 6.4
percentage points of gap detection, and an effect that small cannot clear a
±0.07 seed spread at four seeds per arm. The original experiment was not
looking in the wrong place — it was **underpowered**, and the audit says by
roughly how much.

### 7.1 The audit designs and predicts the powered experiment

16 vs 64 beams spans 0.778 → 0.949 in detection: a 17-point contrast, nearly
three times the 6.4 points of 32 vs 64. Scaling the +0.042 observed at 6.4
points implies roughly +0.11 success.

We pre-registered `narrow` success as the single primary endpoint, six seeds
per arm, exact permutation test, before the runs completed.

| Condition | 16 beams | 64 beams | Δ | p (exact) |
|---|---|---|---|---|
| **narrow** (primary) | 0.597 ± 0.054 | 0.682 ± 0.060 | **+0.085** | **0.035** |
| dense | 0.580 ± 0.035 | 0.695 ± 0.040 | **+0.115** | **0.002** |
| nominal | 0.898 ± 0.041 | 0.937 ± 0.021 | +0.038 | 0.056 |

Observed +0.085 to +0.115, against a predicted ~+0.11. **A training-free
measurement forecast the outcome of a twelve-run training experiment** — a
prediction rather than a fit, and stronger evidence for the mechanism than the
effect size alone.

**On multiplicity:** `narrow` success was pre-registered as the single primary
endpoint, so p = 0.035 stands uncorrected — that is what pre-registration buys.
The eight secondary tests do need correction; at Bonferroni (0.05/8 = 0.006)
only `dense` success and `dense` SPL survive. Stated plainly because p = 0.035
would *not* survive correction if `narrow` were treated as one of nine
exploratory tests, and the only thing separating those readings is having fixed
the endpoint in advance.

## 8. Result 7: field of view beats angular resolution

With resolution established as a real constraint, which property of the sensor
matters? A forward-facing depth camera and a 360° lidar trade off in opposite
directions:

| Sensor | Angular resolution | Resolves 0.44 m gap to | World visible |
|---|---|---|---|
| 64 beams / 360° | 5.62° | 4.5 m | 100% |
| 64 columns / 90° | **1.41°** | **17.9 m** | **25%** |

Both give a 69-dimensional observation to an identical network, so capacity is
matched and the sensor is the only difference. Six seeds per arm, `narrow`
success pre-registered, direction predicted in advance.

| Condition | Lidar 360° | Depth 90° | Δ | p (exact) |
|---|---|---|---|---|
| **narrow** (primary) | 0.682 ± 0.060 | 0.585 ± 0.037 | **−0.097** | **0.019** |
| dense | 0.695 ± 0.040 | 0.565 ± 0.053 | **−0.130** | **0.004** |
| nominal | 0.937 ± 0.021 | 0.862 ± 0.039 | **−0.075** | **0.004** |

Quadrupling angular precision does not come close to paying for losing three
quarters of the view. The penalty **grows with clutter** (−0.075 → −0.097 →
−0.130), the signature of peripheral awareness being the scarce resource.
Collisions are statistically unchanged on `narrow` and `dense`; the camera
policy times out more instead — the same cautious-rather-than-capable failure,
reached more often.

The direction was predicted before the runs, so the §7 saturation argument had
forecasting content. We rate this weaker than §7 nonetheless: a confirmed
*directional* prediction cannot rule out reasoning fitted to an expected
answer, whereas §7 predicted a *quantitative* effect size.

## 9. Result 8: the FOV curve, and a second forecast that held

Section 8 established a direction from two points. Sweeping field of view at a
**fixed 64 samples** turns it into a curve, and separates coverage from
resolution — at 64 samples, widening the view necessarily coarsens it, and
360°/64 columns *is* the 64-beam lidar.

The audit was run first, before any of the new policies existed:

| FOV | Spacing | Gap detection |
|---|---|---|
| 90° | 1.41° | 0.401 |
| 180° | 2.81° | 0.564 |
| 270° | 4.22° | 0.718 |
| 360° | 5.62° | 0.949 |

Coverage dominates completely: quadrupling angular precision while cutting the
view to a quarter more than halves what the sensor can see.

### 9.1 Choosing the test before seeing the data

The audit's numbers, anchored on the two already-trained levels, predicted
adjacent FOV levels would differ by only ~0.03 — **below the ±0.06 seed
spread**. Pairwise tests would therefore have been underpowered by
construction, and running all six of them would have invited reporting
whichever cleared 0.05.

The pre-registered primary was therefore a **monotone trend** across all four
levels — Spearman ρ between FOV and per-seed `narrow` success over 24 seeds,
permutation-tested — chosen from the predicted effect sizes rather than from
the results.

### 9.2 Result

| FOV | Per-seed `narrow` success | Mean ± sd |
|---|---|---|
| 90° | 0.60, 0.65, 0.55, 0.55, 0.58, 0.58 | 0.585 ± 0.037 |
| 180° | 0.79, 0.68, 0.58, 0.55, 0.61, 0.60 | 0.635 ± 0.087 |
| 270° | 0.63, 0.55, 0.66, 0.73, 0.59, 0.69 | 0.642 ± 0.066 |
| 360° | 0.70, 0.58, 0.64, 0.74, 0.71, 0.72 | 0.682 ± 0.060 |

**Spearman ρ = +0.508, Monte Carlo p = 0.0126** (100,000 permutations),
linear slope +0.089 success across 90° → 360°.

Note that no adjacent pair is individually significant — as predicted. The
evidence lives in the ordering across all four levels, which is exactly why
the test was chosen in advance.

### 9.3 Scoring the forecast honestly

| FOV | Predicted | Observed | Error |
|---|---|---|---|
| 90° | 0.585 | 0.585 | — *(anchor)* |
| 180° | **0.614** | **0.635** | **+0.021** |
| 270° | **0.641** | **0.642** | **+0.001** |
| 360° | 0.682 | 0.682 | — *(anchor)* |

The script reports a mean absolute error of 0.006 across all four levels.
**That number flatters the forecast and should not be quoted.** Two of the
four points were the anchors used to fit the slope, so their error is zero by
construction. Only 180° and 270° were genuine out-of-sample predictions, and
their mean absolute error is **0.011** — still small against a ±0.06 seed
spread, and the 270° prediction landed within 0.001, but the honest figure is
roughly twice the one the tooling prints.

This is the second time in the project a training-free measurement forecast
the outcome of a training experiment (cf. §7.1). Two successful quantitative
forecasts from the same audit is the strongest evidence here that the
mechanism is real rather than a story fitted after the fact.

### 9.4 A guess that was wrong

Mid-sweep, validation SPL suggested the curve **saturated** at 180–270°
(0.758, 0.787, 0.792, 0.787) rather than continuing to rise. It does not: the
held-out benchmark increases from 0.642 to 0.682 between 270° and 360°.

Validation SPL is a maximum over ~30 checkpoints measured on the training
distribution — the same statistic that produced a misleading clean separation
in §6.2. It was flagged as unreliable when noted and proved so again. When a
selection statistic and a held-out measurement disagree, the held-out
measurement wins.

## 10. Result 9: coverage is causal, sample count is not

Section 9 held **samples** fixed, so coverage and resolution necessarily moved
together and the curve could not attribute the effect to either. Since
resolution = FOV / samples, both cannot be pinned; decoupling means choosing
which, and pinning **resolution** isolates coverage.

The audit ran first. At a constant 2.81°/sample, detection is almost
indistinguishable from the fixed-64-sample sweep:

| FOV | Fixed 64 samples | Constant 2.81° |
|---|---|---|
| 90° | 0.401 (1.41°) | 0.395 (32 samples) |
| 180° | 0.564 | 0.564 (64) |
| 270° | 0.718 (4.22°) | 0.729 (96) |
| 360° | 0.949 (5.62°) | 0.974 (128) |

Detection is set by FOV alone. Doubling samples buys 0.006 at 90° and 0.025 at
360°.

### 10.1 Pre-registration

Three comparisons, six seeds each, `narrow` success as the primary endpoint in
every case. Crucially, the two predicted nulls were given **explicit magnitude
bounds** rather than the unfalsifiable "not significant":

| Comparison | Holds fixed | Predicted |
|---|---|---|
| 32@90° vs 64@90° | FOV | null, \|Δ\| < 0.01 |
| 64@360° vs 128@360° | FOV | null, \|Δ\| < 0.01 |
| 32@90° vs 128@360° | 2.81°/sample | **+0.10, significant** |

### 10.2 Result

| Comparison | Δ success | p | Predicted | Held? |
|---|---|---|---|---|
| 64@90° vs 32@90° | **+0.002** | 1.000 | null <0.01 | **yes** |
| 128@360° vs 64@360° | **−0.003** | 0.955 | null <0.01 | **yes** |
| **128@360° vs 32@90°** | **+0.095** | **0.024** | +0.10 | **yes** |

`dense` agrees: +0.088, p = 0.028 for the coverage comparison.

**Doubling the sample count changes nothing — twice, at both ends of the FOV
range. Quadrupling coverage at identical angular resolution produces the whole
effect.** Coverage is the causal factor; angular resolution, over this range,
is second-order.

This also resolves §7's loose end. A single-seed run had suggested 128 beams
was *worse* than 64; across six seeds the difference is −0.003. That
regression was seed noise, exactly as the 32-vs-64 case turned out to be.

### 10.3 A secondary that does not survive correction

At 360°, the 128-beam arm collides significantly more than the 64-beam arm
(0.188 vs 0.105, p = 0.006) with success unchanged — a failure-composition
shift rather than a performance change. With roughly nine secondary tests
across the three comparisons, the Bonferroni threshold is 0.0056, so **p =
0.006 does not survive correction.** It is recorded as suggestive and nothing
more.

### 10.4 Why this reframes the earlier results

Sections 8 and 9 were reported as "field of view beats angular resolution".
Section 10 sharpens that: it is not that FOV beats resolution in a trade-off,
but that **resolution barely matters at all** in this range while coverage
carries the entire effect. The earlier framing implied a frontier where one
could buy performance with either knob. There is no such frontier here — one
knob is inert.

## 11. Result 10: what the encoder costs

Every result so far used range data. This one uses images, and is constructed
so that the *only* thing that changes is the representation: the RGB camera
renders the same geometry the depth camera measures, at the same 90° FOV and
the same 64 columns, with the goal vector bit-identical between modes. The
comparison therefore isolates **what it costs to make a CNN recover from
pixels what an MLP reads directly from a vector.**

Six seeds per arm, `narrow` success pre-registered as primary. The prediction,
recorded before the runs: RGB loses by **0.05–0.10**.

### 11.1 Result

| Condition | Depth (MLP) | RGB (CNN) | Δ | p (exact) |
|---|---|---|---|---|
| **narrow** (primary) | 0.585 ± 0.037 | 0.367 ± 0.090 | **−0.218** | **0.002** |
| dense | 0.565 ± 0.053 | 0.327 ± 0.097 | **−0.238** | **0.002** |
| nominal | 0.862 ± 0.039 | 0.700 ± 0.059 | **−0.162** | **0.002** |

All three sit at the 2/924 permutation floor: every depth seed beats every RGB
seed on every condition.

**The prediction was wrong.** The direction held, but the magnitude is two to
three times what was forecast. Unlike the two audit-based forecasts in §7.1
and §9.3, this one had no measurement behind it — it was an intuition, and it
was badly calibrated. The contrast is worth drawing: the forecasts that held
were derived from a quantity that had actually been measured; the one that
failed was a guess dressed in the same confident register.

### 11.2 The encoder also costs reliability

| Arm | `narrow` success sd | `narrow` collision rate |
|---|---|---|
| depth | ±0.037 | 0.117 ± 0.037 |
| RGB | **±0.090** | 0.222 ± **0.160** |

Seed-to-seed spread roughly doubles for success and quadruples for collisions,
with individual RGB seeds ranging from 0.02 to 0.45 collision rate. The CNN
arm is not merely worse on average; it is far less *predictable*, which for a
deployed system is arguably the more serious defect.

### 11.3 Confounds, one of them untested

- **Not a pure encoder ablation.** The RGB arm also changes policy
  architecture (`MultiInputPolicy` + NatureCNN) and batch size (256 vs 512,
  forced by image memory). It measures the cost of moving to pixels *with a
  standard CNN setup*, not the encoder in isolation.
- **Compute has since been tested and rejected** — see §11.5. This was
  flagged here as the least-defended claim in the report, and the sweep was
  run to close it.
- **The render is synthetic and clean.** No texture, lighting, motion blur or
  sensor noise. A real camera is harder, so this is a *lower bound* on the
  encoder's cost.

### 11.4 What it means

The information was held constant and the representation was not, and the
representation cost 0.16–0.24 success. That is a large price for a change that
adds nothing to what the agent can in principle know — and it is paid before
any of the difficulties a real camera introduces.

For a project titled *vision-conditioned navigation*, this is the least
comfortable and most useful result in it: **on this task, the pixels are the
problem, not the geometry.**

### 11.5 The compute excuse, tested and rejected

§11.3 flagged the compute confound as the report's weakest claim. It has now
been tested the same way §5.2 tested it for domain randomisation: the RGB
policies were resumed to **4.0M steps, 2.7× the depth arm's budget**.

| Comparison | Δ `narrow` success | p | Predicted |
|---|---|---|---|
| rgb@4.0M vs rgb@1.5M | +0.057 | 0.524 | +0.00 to +0.05 ✓ |
| **rgb@4.0M vs depth@1.5M** | **−0.162** | **0.030** | still < −0.15 ✓ |

Extra compute does not significantly improve RGB, and **the encoder gap
survives even when RGB is handed 2.7× the compute of the arm it loses to.**
Compute accounts for roughly a quarter of the original 0.218 gap; the
remaining 0.162 does not go away.

One genuinely unexpected secondary: seed variance **increased** with more
training (±0.153 vs ±0.090 on `narrow` success). More compute made the RGB
arm less consistent, not more — the opposite of what convergence looks like.

### 11.6 A methodological lesson that recurs

Mid-sweep, validation SPL rose from 0.584 to 0.659 and this was briefly
recorded as the prediction having failed. It had not: held-out success moved
only +0.057, not significantly.

That is the **third** time in this project that validation SPL has been more
optimistic than the held-out benchmark — in the seed replication (§6.2), in
the FOV saturation guess (§9.4), and here. It is a maximum over ~30
checkpoints measured on the training distribution, so it is both biased upward
and lower-variance than the quantity that matters. Three independent
occurrences is enough to state the rule plainly: **when a selection statistic
and a held-out measurement disagree, the held-out measurement wins, and the
selection statistic should not be read as a preview of it.**

## 12. Result 11: where the map is wrong — the gap shrinks but does not reverse

Every condition so far hands the classical planner a **perfect, current,
static map**. That is its largest privilege and the one real deployments do
not have: maps go stale and obstacles move. This condition adds obstacles that
move and are **absent from the map**, which is the first place a reactive
policy has a structural reason to win.

**Hypothesis, recorded before the runs:** the learned policy would beat the
classical planner here.

### 12.1 Keeping the baseline honest

Nav2 does not plan once and drive blind; it maintains a local costmap from
live sensing and replans continuously. So the baseline replans against a
costmap containing the movers where they currently are.

Sweeping that interval turned out to matter, and not in the direction
expected:

| Condition | no replan | replan 10 | replan 25 |
|---|---|---|---|
| nominal | **1.000** | 0.960 | 0.990 |
| narrow | **0.850** | 0.690 | 0.760 |
| dynamic | 0.810 | **0.870** | 0.860 |
| dynamic_dense | 0.680 | **0.750** | 0.730 |

Replanning is worth +0.06 to +0.07 where movers exist and costs up to −0.16
where they do not, through path churn in tight corridors. **A single global
setting would have handicapped the baseline on half the suite.** The
comparison therefore gives the classical planner its best configuration on
every condition.

This nearly produced a false result. With `replan_every=10` applied
everywhere, `narrow` showed classical at 0.690 against the learned policy's
0.682 — 4 of 6 seeds above the baseline, which reads as parity. That parity
was entirely an artefact of a handicap introduced in the name of *fairness*
one step earlier. It survived only because the interval was swept rather than
assumed.

### 12.2 Result

Success rate, 100 held-out worlds, six seeds per learned arm, classical at its
best configuration:

| Condition | Classical | Lidar 360° | Δ | sign test |
|---|---|---|---|---|
| nominal | **1.000** | 0.937 ± 0.021 | −0.063 | 0/6, p = 0.031 |
| narrow | **0.850** | 0.682 ± 0.060 | **−0.168** | 0/6, p = 0.031 |
| dynamic_dense | **0.750** | 0.700 ± 0.023 | −0.050 | 0/6, p = 0.031 |
| **dynamic** | **0.870** | 0.848 ± 0.019 | **−0.022** | 0/6, **p = 0.062** |

**The hypothesis is falsified.** The learned policy does not beat the
classical planner even where the map is wrong. No seed, on any condition,
exceeds the baseline.

### 12.3 What is real

The gap **shrinks roughly eightfold**: −0.168 in tight static corridors versus
−0.022 with movers. And `dynamic` is the only condition in the entire study
where classical's advantage is **not statistically established** — p = 0.062,
above 0.05, with one seed exactly tying the baseline.

So the honest statement is narrower than the hypothesis and still worth
having: *reactive control closes most of the distance precisely where the map
degrades, without overtaking.* The direction of the effect is exactly what was
predicted; only the magnitude falls short of a reversal.

It is also the first two-sided result in the project. Everywhere else the
answer was "classical wins, and here is why"; here it is "classical wins by an
amount too small to demonstrate."

### 12.4 Why this is a weaker test than it looks

- **The movers are benign.** They oscillate on fixed line segments at
  0.15–0.45 m/s and never pursue the robot. Real pedestrians are faster, less
  predictable, and reactive. A harder mover distribution is the obvious way to
  push this further.
- **The policies are zero-shot.** None was trained with moving obstacles,
  which makes the near-parity more impressive, but also means a policy trained
  on movers was never tested — the natural next experiment and the one most
  likely to produce an actual reversal.
- **The classical planner still gets a perfect static map** and exact pose.
  Only the movers are hidden from it. Staleness in the *static* layout would
  be a further, more realistic degradation.

## 13. Result 12: motion information does not help, but parity arrives anyway

§12 found the learned policy did not win where the map is wrong. Diagnosing
*why* suggested the test had been unfair to it: **a single range scan carries
obstacle positions but no velocities.** A one-frame reactive policy is in
exactly the same epistemic position as a replanning planner — both must treat
a mover as a static obstacle wherever it currently sits. The policy had no
informational advantage to exploit.

Frame stacking breaks that symmetry. Two scans encode motion; four give a
cleaner velocity estimate. This is the only configuration in the study where
the learned policy holds information the classical stack structurally lacks.

**Prediction, recorded before the runs:** the 4-frame policy beats classical
by +0.01 to +0.05.

### 13.1 Design

Two arms, six seeds each, both trained *with* movers in the distribution:

| Arm | Frame stack | Role |
|---|---|---|
| `dyn1` | 1 | control — isolates mover exposure from motion information |
| `dyn4` | 4 | the mechanism under test |

The control matters: without it, any win could not be attributed to seeing
motion rather than simply having trained on movers.

### 13.2 Result

Success rate, 100 held-out worlds, classical at its best configuration:

| Condition | Classical | `dyn1` | `dyn4` |
|---|---|---|---|
| dynamic | 0.870 | 0.860 ± 0.033 | 0.852 ± 0.038 |
| dynamic_dense | 0.750 | 0.697 ± 0.060 | 0.710 ± 0.042 |

**The prediction is falsified.** `dyn4` does not beat classical
(−0.018, 2/6 seeds above, p = 1.000).

**And the mechanism did nothing.** Frame stacking versus its control:

| Condition | `dyn4` − `dyn1` | p (exact permutation) |
|---|---|---|
| dynamic | −0.008 | 0.784 |
| dynamic_dense | +0.013 | 0.703 |

Motion information — the one advantage the learned side structurally had —
produced no measurable benefit in either direction. Training on movers at all
was likewise negligible: `dyn1` versus the zero-shot policy is +0.012,
p = 0.541.

### 13.3 What did happen

Neither trained arm is *significantly worse* than the classical planner on
either dynamic condition: p = 0.625 and 1.000 on `dynamic`, 0.219 and 0.125 on
`dynamic_dense`. Set against `narrow`, where the same lidar policies lose by
−0.168 at p = 0.031, the contrast is clean:

| Regime | Δ vs classical | Significant? |
|---|---|---|
| static clutter (`narrow`) | −0.168 | **yes**, p = 0.031 |
| moving obstacles (`dynamic`) | −0.010 | no, p = 0.625 |
| moving obstacles, cluttered | −0.053 | no, p = 0.219 |

So the honest claim is **parity, not victory**: where the map is wrong the
learned policy becomes statistically indistinguishable from a strong
replanning planner, while remaining clearly worse where the map is right. The
regime change is real; the reversal never arrives.

### 13.4 Why the mechanism probably failed

Offered as hypotheses, not findings — none is tested here:

- **The movers may be too slow to matter.** At 0.15–0.45 m/s against a robot
  capped at 0.6 m/s, avoiding them reactively may simply be easy enough that
  anticipation buys nothing.
- **Velocity may be hard to extract from raw stacked scans.** An MLP given
  four concatenated 69-d vectors must learn differencing itself; an explicit
  velocity feature, or a recurrent policy, might succeed where this does not.
- **The 500-step budget rewards caution.** As §5.3 established, this reward
  makes stalling preferable to collision by 4:1, so a policy that could
  anticipate might still choose to wait rather than commit.

The first is the cheapest to test and would be the next step: faster movers
should widen any anticipation advantage, if one exists at all.

## 14. Discussion

**A learned policy did not beat a strong classical planner on this task, and
the reasons are now specific rather than vague.** Four standard explanations
were tested and eliminated. What remains is a genuine constraint (sensor
geometry) and a genuine behavioural finding (the policy optimises the reward
correctly, and that reward prefers stalling to crashing by 4:1).

Two asymmetries in the comparison both **favour** the learned side, and it
still lost: the classical planner has no training distribution, so the shifts
are not shifts for it; and the randomised policy received 2.7× the compute.

**What the learned policies do win**, both small and caveated: ~6% fewer steps
on successful nominal episodes (mildly flattered by survivorship), and
indifference to 0.10 m range noise — the one axis on which the classical
baseline cannot be compared at all, since it never reads the sensor.

**The transferable lesson is methodological.** A correctly computed
significance test produced a confident, reproducible, wrong conclusion because
it measured episode variance rather than seed variance. The tooling that caught
it — seed as unit of analysis, exact permutation tests, pre-registered
endpoints, and training-free mechanism measurement — is cheap and should be
default practice.

## 15. Limitations

- **Simulation is 2D and analytic.** No dynamics, no sensor artefacts beyond
  additive noise and dropout, no appearance. Conclusions about *geometry* should
  transfer; conclusions about perception in the full sense should not be assumed
  to.
- **Sample sizes are small.** Six seeds per arm is enough to detect ~0.09
  against a ±0.06 spread, and not enough to characterise a curve. The 128-beam
  regression relative to 64 remains unexplained and unreplicated.
- **One camera configuration.** §8 establishes a direction, not a frontier. A
  180°/270° camera, or one with memory across frames, might close the gap. The
  FOV-vs-resolution trade has two knobs and this samples one point.
- **Reward is not exhaustively searched.** §5.4 tested three single-term
  changes, not the joint space.
- **The classical baseline is not Nav2.** It is structurally analogous and
  strong on this task (1.000 nominal success), but a real Nav2 comparison would
  be more convincing.
- **Single-seed findings remain flagged as unreplicated**, including the one
  significant reward-ablation result (`step_penalty` improving nominal SPL by
  +0.066).

## 16. Future work

In order of expected information per GPU-hour:

1. **FOV sweep at fixed column count** (90/180/270/360°). Converts §8 from a
   point into a curve and separates FOV from resolution, which §8 deliberately
   confounds.
2. **Frame stacking or recurrence.** Distinguishes *missing* information from
   *forgetting* it — the two predict different outcomes.
3. **RGB observations with a CNN encoder**, testing whether an image encoder
   inherits the same angular-resolution constraint.
4. **Nav2 as the baseline**, over the same task via ROS 2, removing the
   "structurally analogous" caveat.
5. **Sim-to-real** on a TurtleBot-class base. The action space is already
   `Twist`, so the policy transfers without modification.

## 17. Reproducing

```bash
pip install -e ".[dev,viz]"
pytest                                        # 161 tests
python -m vision_nav.training.train           # privileged RL
python scripts/run_benchmark.py --rl <model>  # comparison matrix
python scripts/perception_audit.py            # §7, no training required
python scripts/seed_analysis.py --arm ...     # §6.2, §7.1, §8
```

Every training run writes its fully resolved config alongside its checkpoints,
so any number here is traceable to the settings that produced it.

---

## References

1. P. Anderson et al. *On Evaluation of Embodied Navigation Agents.* arXiv:1807.06757, 2018.
2. P. Henderson et al. *Deep Reinforcement Learning That Matters.* AAAI, 2018.
3. J. Schulman et al. *Proximal Policy Optimization Algorithms.* arXiv:1707.06347, 2017.
4. S. Macenski et al. *The Marathon 2: A Navigation System.* IROS, 2020. (Nav2)
5. J. Tobin et al. *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World.* IROS, 2017.
6. A. Raffin et al. *Stable-Baselines3: Reliable Reinforcement Learning Implementations.* JMLR, 2021.
