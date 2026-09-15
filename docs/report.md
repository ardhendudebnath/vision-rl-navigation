# Where Learned Navigation Loses to Classical Planning, and Why

**A controlled comparison of reinforcement-learning and search-based navigation
under distribution shift**

Ardhendu Debnath — [vision-rl-navigation](https://github.com/ardhendudebnath/vision-rl-navigation)

---

## Abstract

We compare a PPO navigation policy against a classical A\* + pure-pursuit
stack on a procedurally generated point-goal task, across five environment
distributions, with the planner given the full map and exact pose throughout.
**The planner wins on every condition**, by margins that grow with clutter
(1.000 vs 0.960 success nominally; 0.850 vs 0.682 in tight corridors).

We then eliminate the standard explanations for a losing RL result —
insufficient data, wrong training distribution, insufficient compute (tested
twice, to 2.7× budget), and reward mis-specification — and find the cause is
behavioural: the reward makes a collision cost four times a timeout, so the
policy correctly learns to stall rather than crash.

Turning to perception, a **training-free audit of the sensor** quantifies a
real geometric limit and then forecasts two subsequent training experiments to
within 0.021 and 0.001. It shows coverage is causal while angular resolution
is inert: doubling sample count at fixed field of view changes nothing
(±0.003, inside a pre-registered ±0.01 bound), while quadrupling coverage at
identical resolution produces the whole effect (+0.095, p = 0.024). Holding
information constant and changing only the *representation* — the same
geometry as pixels for a CNN rather than a vector for an MLP — costs 0.16–0.24
success. Where the map is wrong the gap narrows to statistical parity with
sparse movers (−0.020, p = 0.219) but never reverses, and a controlled
subtraction that freezes those movers shows why: the classical advantage
returns in full (−0.100, −0.202, both p = 0.031), so the parity is motion
degrading the planner (−0.120 to −0.160) rather than the policy handling it
(−0.048 to −0.068).

A reward ablation sharpens what "behavioural" means: the same 4:1 ratio also
sets what *extra information* is worth. Frame stacking cuts collisions under
either reward, but at 4:1 the saving is spent on timeouts and net success
falls, while at a 1:1 reward it becomes successes (+0.033, p = 0.019, 6/6
seeds). An observation channel is worth only what the objective lets the
policy do with it.

**The most transferable contribution is methodological.** A correctly computed
significance test produced a confident, reproducible, and wrong conclusion,
because it measured episode variance rather than training-seed variance. We
document that failure in full, alongside the measurement that diagnosed it.

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
tracking that plan. That symmetry was deliberate, so that Nav2 itself could
replace the baseline without changing the comparison protocol. Section 4.1
does exactly that, using Nav2 1.3.12 on ROS 2 Jazzy (Macenski et al., 2020).

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

### 4.1 Nav2 as the baseline

If the hand-written stack happened to be weak, "the learned policy loses to
classical planning" would shrink to "it loses to one particular script". So
Nav2 1.3.12 (ROS 2 Jazzy) runs as one more actor over a ROS 2 bridge — same
worlds, same seed order, same metrics, same action space — and is given the
generous side of every choice: ground-truth pose, the static map, and a
360-beam scan, 3–22× denser than any learned policy receives. Setup and the
three silent failure modes the harness guards against are in
[`ros2_bridge/README.md`](../ros2_bridge/README.md).

Nav2 is not deterministic — it is a set of asynchronous processes, and timing
jitter changes which trajectory DWB selects. Measured run-to-run spread is
0.000–0.030 success. Every other actor here is deterministic or replicated
over six seeds; Nav2 is neither, so it is reported as a range over two passes
and a winner is called only when both passes fall the same side of the noise
band.

| Condition | Hand-written | Nav2 (2 passes) | Δ success |
|---|---|---|---|
| nominal | 1.000 / 0.985 | 0.970–0.980 / 0.955–0.963 | −0.030 to −0.020 |
| sparse | 1.000 / 1.000 | 0.990 / 0.982–0.985 | −0.010 |
| large | 1.000 / 0.990 | 0.990 / 0.983–0.985 | −0.010 |
| noisy_lidar | 1.000 / 0.985 | 0.970–0.980 / 0.953–0.964 | −0.030 to −0.020 |
| dense | 0.890 / 0.841 | 0.910–0.940 / 0.881–0.901 | +0.020 to +0.050 |
| **narrow** | 0.850 / 0.795 | **0.910–0.930** / **0.878–0.898** | +0.060 to +0.080 |

**The hand-written baseline understates classical planning in tight
corridors.** On the four conditions where clutter is not the binding
constraint the two agree to within 0.03 — Nav2 marginally behind, which is
what a general-purpose stack against a baseline tuned for this exact task
should look like. On `narrow` Nav2 is ahead in both passes by more than the
run-to-run spread. On `dense` it is ahead in both passes too, but by
+0.020 to +0.050, which straddles the noise band; that one is suggestive, not
established.

This does not weaken Result 1 — it widens it, on `narrow`. The learned
policies were losing to the weaker of the two classical stacks, so the
published gap there is a lower bound.

**The mechanism is the one this report already diagnosed, and it is cleaner
than the success rates.** Section 4 measured that A\* never once fails to find
a route, and that 9/11 (`dense`) and 12/15 (`narrow`) of the baseline's
failures are the *controller* losing the plan while cornering. DWB re-scores
obstacle-aware trajectories every control cycle instead of tracking a fixed
path, and collisions fall accordingly:

| Condition | Hand-written collisions | Nav2 collisions | Reduction |
|---|---|---|---|
| narrow | 0.120 | 0.010–0.050 | **58–92%** |
| dense | 0.090 | 0.020–0.040 | **56–78%** |

On the four uncluttered conditions both stacks collide at 0.000–0.020 and the
change is nil — collisions fall only where the controller was the binding
constraint, which is the prediction those failure-mode counts licensed. Where
the recovered episodes go differs: on `narrow` they become successes, on
`dense` mostly timeouts (0.020 → 0.040–0.050), which is why the collision fix
does not convert into a success gain that clears the noise there. **Fixing the
diagnosed failure mode is not the same as fixing the outcome.**

**One caveat, because it nearly produced the opposite result.** Nav2's
costmaps are stateful: obstacle marks clear only by ray-tracing from the
current pose, and the robot teleports between episodes, so the global costmap
silted up with geometry from previous worlds until the planner could not find
a route. Episodes were not independent, though every other actor's are. The
tell was a diagnostic unrelated to the outcome — episodes where Nav2 issued
*zero* velocity commands (12% of `large`, 8% of `dense`), because success rate
alone cannot separate a robot that navigates badly from one that never moves.
Before the fix this experiment reported Nav2 as **worse** on `dense` by 0.130,
in the direction that flattered this project's own baseline.

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
failures. On `narrow`, collisions fall from 27 per 100 episodes to 2 as the
policy is given randomisation and 2.7× the compute — it has all but learned
not to crash. But of the 25 episodes that left the collision bucket, **21
became timeouts and only 4 became successes**, and the stalled ones crawl at
0.14 m/s against a 0.6 m/s cap.

Measured episode returns explain why: success +41.20, timeout −2.06, collision
−24.91. A collision costs a flat −20; timing out for all 500 steps costs −5.
**Crashing is four times worse than stalling forever.** The policy is not
malfunctioning — it found the optimum of the reward it was given, and that
reward asks it to learn not to crash rather than how to get through. Full
failure-composition and return tables are in [Phase 2e](project_plan.md).

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

That is the whole story on a static task. It is not the whole story about the
reward: §9.1 finds the same ratio decides whether *extra information* is worth
anything, converting a frame-stacking collision saving into timeouts at 4:1
and into successes at 1:1. The reward does not close the gap to the planner —
that is what this section rules out — but it does set the exchange rate
between information and performance.

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

16 vs 64 beams spans 0.778 → 0.949 in detection — a 17-point contrast against
the 6.4 points of 32 vs 64 — so scaling the +0.042 observed at 6.4 points
implies roughly **+0.11 success**. `narrow` success was pre-registered as the
single primary endpoint, six seeds per arm, exact permutation test, before the
runs completed.

Observed: **+0.085 on `narrow` (p = 0.035)** and +0.115 on `dense`
(p = 0.002), against that predicted +0.11. **A training-free measurement
forecast the outcome of a twelve-run training experiment** — a prediction
rather than a fit, and stronger evidence for the mechanism than the effect
size alone. It also settles §6: the beam-count effect is real, and §6.1 was
underpowered rather than wrong about where to look.

**On multiplicity:** the primary endpoint was fixed in advance, so p = 0.035
stands uncorrected — that is what pre-registration buys. The eight secondary
tests do need it, and at Bonferroni only `dense` success and SPL survive.
Stated plainly because p = 0.035 would *not* survive correction if `narrow`
were treated as one of nine exploratory tests. Per-seed values and the full
table are in [Phase 2i](project_plan.md).

## 8. Results 7–10: what the sensor and its representation cost

§7 established that sensor geometry is a real constraint. Four further
experiments, each pre-registered with six seeds per arm, establish *which*
property matters and what it costs to change the representation.

| Experiment | Comparison | Δ `narrow` success | p |
|---|---|---|---|
| Depth vs lidar | 64 col @ 90° vs 64 beams @ 360° | −0.097 | 0.019 |
| FOV sweep | 90→360° at 64 samples, 24 seeds | ρ = +0.508 | 0.013 |
| Samples at fixed FOV | 32 vs 64 @ 90°; 64 vs 128 @ 360° | +0.002, −0.003 | 1.000, 0.955 |
| Coverage at fixed resolution | 32 @ 90° vs 128 @ 360°, both 2.81°/sample | **+0.095** | **0.024** |
| Representation | RGB + CNN vs depth + MLP | **−0.218** | **0.002** |

### 8.1 Coverage is causal; resolution is not

A depth camera with **4× finer** angular sampling but a 90° field of view
loses to a 360° lidar on every condition. Sweeping field of view at a fixed
sample count turns that into a monotone curve.

The decisive test pins *resolution* instead and varies coverage — the two
cannot both be held, since resolution = FOV/samples. **Doubling the sample
count changes nothing, twice, at both ends of the range. Quadrupling coverage
at identical angular resolution produces the entire effect.** This sharpens
the earlier framing rather than confirming it: "field of view beats angular
resolution" implies a frontier where either knob buys performance. There is no
such frontier — one knob is inert.

### 8.2 The audit forecast both results before the policies existed

The perception audit of §7 predicted the FOV sweep's two interior levels to
within **+0.021 and +0.001**, from a measurement taken with no training and no
seeds. (The tooling reports a mean absolute error of 0.006 across all four
levels; that figure flatters the forecast, since two points were the anchors
used to fit the slope. The honest out-of-sample figure is 0.011.)

Two independent quantitative forecasts from the same training-free measurement
is the strongest evidence in this report that the mechanism is real rather
than fitted after the fact.

### 8.3 The pixels are the problem, not the geometry

The RGB camera renders the *same* geometry the depth camera measures — same
90° FOV, same 64 columns, goal vector bit-identical between modes. Only the
representation changes: pixels a CNN must interpret, rather than a vector an
MLP reads directly.

It costs **0.16–0.24 success on every condition**, with every depth seed
beating every RGB seed, and the encoder costs *reliability* too — seed spread
roughly doubles for success and quadruples for collisions. Giving RGB **2.7×
the compute** does not close it (−0.162, p = 0.030). The render is clean — no
texture, lighting or sensor noise — so this is a lower bound.

**A prediction that failed.** I forecast −0.05 to −0.10; the effect is two to
three times that. Worth contrasting with §8.2: the forecasts that held were
derived from a measured quantity, this one was intuition in the same confident
register.

## 9. Results 11–12: where the map is wrong

Every condition so far hands the classical planner a **perfect, current,
static map** — its largest privilege and the one real deployments lack. Adding
obstacles that move and are absent from the map is the first place a reactive
policy has a structural reason to win.

**It does not win.** Two hypotheses were pre-registered and both failed: that
zero-shot policies would beat the planner, and that frame stacking — which
gives the policy velocity information the map-based stack structurally cannot
have — would tip it.

| Condition | Classical | Learned (best) | Δ | Significant? |
|---|---|---|---|---|
| narrow (static clutter) | **0.850** | 0.682 ± 0.060 | −0.168 | **yes**, p = 0.031 |
| dynamic | **0.880** | 0.860 ± 0.033 | −0.020 | no, p = 0.219 |
| dynamic_dense | **0.820** | 0.710 ± 0.042 | −0.110 | **yes**, p = 0.031 |

The classical rows use block-triggered replanning, its best configuration on
every dynamic condition ([`dynamic_obstacles.md`](dynamic_obstacles.md)). Against the timed baseline published earlier
they read 0.870 and 0.750, and `dynamic_dense` looked like parity (−0.040,
p = 0.125). It is not.

Frame stacking versus its own control: −0.008 (p = 0.784) and +0.013
(p = 0.703). The one structural advantage available produced nothing *under
this reward* — §9.1 shows it produces something under a different one,
which is the more interesting result.

**What is real is a regime change, not a reversal**, and only on `dynamic`.
§9.1 shows the parity is the planner degrading rather than the policy coping.

### 9.1 What the parity is, in one page

Chasing that parity took eight phases and produced three successive
corrections to the same claim. The argument is in
[`dynamic_obstacles.md`](dynamic_obstacles.md); the conclusions are these.

**The parity is the planner degrading, not the policy coping.** Re-running
each condition with the movers *parked* — identical worlds and seeds, movers
still absent from the map, only the motion removed — restores the classical
advantage in full: −0.100 on sparse and −0.202 on dense, 0 of 6 seeds above
the baseline in both (p = 0.031). Motion costs the planner 0.120–0.160 and the
learned policy only 0.048–0.068. The policy is behind in every regime; it is
simply harder to disrupt, because one that never commits to a path has no plan
to invalidate.

**The motion cost itself is unexplained.** Four candidate mechanisms were
tested and eliminated: replanning churn (real, but a clutter pathology that
leaves the cost unchanged), planning failure (A\* never fails to find a
route), sensing (the movers are fully visible), and commitment length (a 6×
sweep of the controller's rollout horizon moves the cost by 0.040, inside
noise, while moving absolute performance by 0.220).

**The reward sets what information is worth.** Frame stacking looked inert
across two mover speeds and three encodings. It is not: the reward prices a
collision at 20 and a full timeout at 5, and making the two equal turns
stacking into a significant gain (+0.033, p = 0.019, 6 of 6 seeds) that is
absent on a slow control. Stacking cuts collisions under *either* reward — the
information was always being used — but at 4:1 the saving is spent on timeouts
and net success falls, while at 1:1 it becomes successes.

**And the baseline needed correcting twice.** Real Nav2 beats the hand-written
stack under clutter (§4.1), and the hand-written stack beats the learned
policy on `dynamic_dense` once given its best replanning policy. Every
narrowing of the headline came from improving the baseline; none came from new
evidence about the policy.

## 10. Discussion

**A learned policy did not beat a strong classical planner on this task, and
the reasons are now specific rather than vague.** Four standard explanations
were tested and eliminated. What remains is three substantive findings:

- **Behavioural.** The policy optimises its reward correctly, and that reward
  prefers stalling to crashing by 4:1. It learns not to crash rather than how
  to get through.
- **Perceptual.** Sensor *coverage* is causal and angular *resolution* is
  inert over the range tested — a distinction the obvious framing
  ("field of view beats resolution") gets wrong, because it implies a frontier
  where either knob buys performance.
- **Representational.** Holding information constant and changing only the
  encoding costs 0.16–0.24 success. On this task the pixels are harder than
  the geometry, before any of the difficulties a real camera adds.

Two asymmetries in the comparison **favour** the learned side, and it still
lost: the classical planner has no training distribution, so the shifts are
not shifts for it; and two learned arms received 2.7× the compute.

**The reward decides what information is worth.** Section 5.3 found the
4:1 collision-to-timeout ratio makes the policy stall rather than get through.
Section 9.1 finds the same ratio also decides what *extra* information buys:
frame stacking cuts collisions under either reward, but at 4:1 the saving is
swallowed by timeouts and net success falls, while at indifference it becomes
successes. An observation channel is worth only what the objective lets the
policy do with it — which is worth knowing before concluding that a sensor or
a representation is useless.

**Where the learned side earns its keep** is narrow and not what it first
appeared. It is indistinguishable from *both* classical stacks on `dynamic`,
~6% faster on successful nominal episodes, and indifferent to sensor noise —
the one axis the hand-written baseline cannot be compared on at all, since it
never reads the sensor. But §9.1 shows that parity is the planner degrading
under motion, not the policy handling it: the policy is behind in every regime
and merely harder to disrupt, because one that never commits to a path has no
plan to invalidate. **Robustness by absence of commitment is a real property
and not the one "RL competes when the map is wrong" implies.**

**The transferable lesson is methodological.** A correctly computed
significance test produced a confident, reproducible, wrong conclusion because
it measured episode variance rather than seed variance. The tooling that
caught it — seed as the unit of analysis, exact permutation tests,
pre-registered endpoints with magnitude bounds on predicted nulls, and
training-free mechanism measurement — is cheap and should be default practice.

**Calibration.** Of nine predictions made in advance, the two derived from a
*measurement* held, to within 0.021 and 0.001; six from extrapolation or
intuition failed outright; the ninth got its mechanism right and its magnitude
wrong (+0.033 against a predicted +0.05). Confidence of expression was
identical throughout. Two failures sharpened the rule rather than breaking it.

Predicting Nav2 ≥ 0.92 on `dynamic` extrapolated a *measured* collision
reduction, which by the rule above should have been reliable — but it crossed
from static clutter to moving obstacles, a boundary the measurement never
spanned. **Measurement-derived predictions hold within the regime measured and
become intuition outside it.**

The churn and horizon experiments each proposed a mechanism, each named a
control cell where that mechanism should not act, and each was refuted by the
control moving as much as the treatment. Without those cells, +0.070 on
`dynamic_dense` and a tidy horizon optimum would both have read as clean
confirmations of explanations the data refutes. **A mechanism claim needs a
cell where the mechanism should not act**, and a story fitting every number
available is weak evidence until one exists. Each cost one extra condition.

**And every narrowing of the headline came from the baseline.** The project's
most RL-favourable result now holds on `dynamic` alone, having been reduced
three times — by a production stack, by freezing the movers, and by giving the
original baseline a better replanning policy. Not once by new evidence about
the policy. A fourth such artefact was caught before publication by sweeping
a parameter instead of assuming it. **When a result favours the thing you are
studying, the baseline is the first place to look — and one pass at it is not
enough.**

## 11. Limitations

- **Simulation is 2D and analytic.** No dynamics, no appearance, no sensor
  artefacts beyond additive noise and dropout. Conclusions about *geometry*
  should transfer; conclusions about perception in the full sense should not
  be assumed to.
- **The RGB render is clean** — no texture, lighting or motion blur — so the
  encoder cost is a **lower bound**, not an estimate.
- **The movers are benign.** Fixed line segments, 0.15–0.45 m/s, never
  pursuing the robot. That is a weak test of anticipation and the most likely
  reason frame stacking showed nothing.
- **Sample sizes are small.** Six seeds per arm resolves ~0.09 against a ±0.06
  spread; it cannot characterise a curve. The 128-beam regression relative to
  64 remains unexplained.
- **Reward is not exhaustively searched.** Three single-term changes, not the
  joint space.
- **The hand-written baseline understates classical planning in tight
  corridors.** Section 4.1 measures it against real Nav2: the two agree within
  0.03 where clutter is not binding, but Nav2 is ahead on `narrow` in both
  passes. The `narrow` gap quoted against `classical` is therefore a *lower
  bound* on the gap to a production stack. The same is probably true of
  `dense`, where both passes favour Nav2 but the margin does not clear the
  noise band.
- **Nav2 itself is measured over only two passes.** It is nondeterministic
  (Section 4.1) and two passes bound its run-to-run spread rather than
  estimating it. The comparison is read against that range, but a tighter
  claim needs more passes. This applies to the dynamic conditions as well, where the
  `dynamic_dense` margin (+0.150 to +0.170) is far outside that spread but the
  `dynamic` result (−0.020 to +0.010) sits inside it and is read as parity
  rather than as a measured equality.
- **Single-seed findings are flagged as unreplicated** throughout, including
  the one significant reward-ablation result (`step_penalty` improving nominal
  SPL by +0.066).

## 12. Future work

In order of expected information per GPU-hour:

1. **Explain the motion cost**, now that four mechanisms are eliminated
   (§9.1). The next candidate worth a controlled test is *velocity in the
   costmap*: every actor here treats each scan as a static snapshot, so none
   can distinguish a mover approaching from one receding. A costmap layer
   carrying per-cell velocity would separate "the world changed" from "the
   world is unknowable", and unlike the four already tested it predicts an
   asymmetry between head-on and crossing movers that is directly checkable.
2. **Re-price the observation results at 1:1.** §9.1 found that the reward
   decides what extra information is *worth*: frame stacking cuts collisions
   under either reward, but at 4:1 the saving is spent on timeouts instead of
   successes. Every other observation-side result in this report — the FOV
   sweep, the resolution null, the depth and RGB encoder costs — was measured
   under that same 4:1 reward. Each is therefore a statement about what the
   objective let the policy do with a channel, not about the channel itself.
   Re-running even one at indifference would show whether "coverage is causal
   and resolution is inert" is a fact about perception or about pricing. This
   is the cheapest way to find out how far Result 2 reaches, and it puts the
   report's own perception findings at risk, which is why it is worth doing.
3. **Recurrence.** The other half of the original item 3, still untested.
   Explicit velocity features are now known to be inert, but a recurrent
   policy could integrate over a longer history than any fixed stack.
4. **Harder perception** — texture, lighting variation, sensor artefacts — to
   turn the encoder-cost lower bound into an estimate.
5. **Sim-to-real** on a TurtleBot-class base. The action space is already
   `Twist`, so the policy transfers without modification.

## 13. Reproducing

```bash
pip install -e ".[dev,viz]"
pytest                                        # 265 tests
python scripts/check_docs.py                  # every doc link resolves
python -m vision_nav.training.train           # privileged RL
python scripts/run_benchmark.py --rl <model>  # comparison matrix
python scripts/perception_audit.py            # §7, no training required
python scripts/seed_analysis.py --arm ...     # §6.2, §7.1, §8
```

Section 4.1 needs ROS 2 Jazzy and Nav2, which live in a userspace conda
environment rather than the project venv (`ros2_bridge/README.md`):

```bash
bash ros2_bridge/run_nav2.sh --condition narrow --episodes 100
python scripts/nav2_comparison.py --nav2-runs results/nav2_runs/*
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
