# Where Learned Navigation Loses to Classical Planning, and Why

**A controlled comparison of reinforcement-learning and search-based navigation
under distribution shift**

Ardhendu Debnath — [vision-rl-navigation](https://github.com/ardhendudebnath/vision-rl-navigation)

---

> **Correction, 6–8 October 2026: until then, every Nav2 result on `noisy_lidar`
> had been measured on a clean sensor.** The ROS bridge built its scanner from the
> condition's settings, so the 0.10 m of range noise was *configured* — which is
> what the fairness check in [`ros2_bridge/README.md`](../ros2_bridge/README.md)
> looked at, and it passed — and then called that scanner without a random
> generator, and the sensor adds noise only when it is handed one. The
> hand-written stack read the noisy sensor throughout; Nav2, with and without
> SLAM, never did. Four runs were affected — §4.1's two full-privilege passes and
> §9.4's two SLAM arms — and all four have been repeated with the noise delivered
> and measured, under a registration committed first (§9.20). The `noisy_lidar`
> cells marked ‡ below are the corrected ones. No pooled verdict changes. One
> conclusion does: §9.19 had slam_toolbox localising three times better "from the
> same odometry and the same scans", and on the same scans it localises no better
> than this stack (0.220 m against 0.228 m, p = 0.52).

## Abstract

We compare a PPO navigation policy against a classical A\* + pure-pursuit
stack on a procedurally generated point-goal task, across five environment
distributions, with the planner given the full map and exact pose throughout.
**The planner wins on every condition**, by margins that grow with clutter
(1.000 vs 0.960 success nominally; 0.850 vs 0.682 in tight corridors) — while
it is handed the map. Building that map from the same scan the policy reads
costs it the clutter margin entirely (0.590 vs 0.600 in tight corridors) and
almost nothing in open worlds. Taking its pose too — odometry, corrected by
scan matching against that same self-built map — costs nothing further in
clutter (0.630 in tight corridors) and most of what is left in open worlds
(0.530 on the largest arenas, against 0.990 with the pose given). The two
privileges are worth opposite things, and structure is why. A production
stack — Nav2 with SLAM — pays at least as much given the same sparse
scanner, and under a fifth of it given a dense one; given that dense scanner
too, this project's stack halves its cost, all of it in localisation. The
pose was standing in for the sensor. The map was standing in for the
implementation: in clutter and under noise, this stack loses 0.14 to 0.22 more
than Nav2 with slam_toolbox — and under noise that is not localisation, where the
two stacks end episodes equally far from the truth.

We then eliminate the standard explanations for a losing RL result —
insufficient data, wrong training distribution, insufficient compute (tested
twice, to 2.7× budget), and reward mis-specification — and find the cause is
behavioural: the reward makes a collision cost four times a timeout, so the
policy correctly learns to stall rather than crash.

Turning to perception, a **training-free audit of the sensor** quantifies a
real geometric limit and then forecasts two subsequent training experiments to
within 0.021 and 0.001. It shows coverage is causal while angular resolution is
inert *in success*: doubling sample count at fixed field of view changes
nothing (±0.003, inside a pre-registered ±0.01 bound), while quadrupling
coverage at identical resolution produces the whole effect (+0.095, p = 0.024).
Re-pricing both under an indifferent reward separates them: coverage's effect
survives unchanged, resolution's reverses sign. Holding information constant
and changing only the *representation* — the same geometry as pixels for a CNN
rather than a vector for an MLP — costs 0.16–0.24 success, and an indifferent
reward leaves that cost standing. Where the map is wrong the gap narrows to
statistical parity with sparse movers (−0.020, p = 0.219) but never reverses,
and a controlled subtraction that freezes those movers shows why: the classical
advantage returns in full (−0.100, −0.202, both p = 0.031), so the parity is
motion degrading the planner (−0.120 to −0.160) rather than the policy handling
it (−0.048 to −0.068). Given the movers' exact trajectories, a planner
reasoning in space-time with a small temporal margin removes that planner cost
almost entirely; a constant-velocity estimate in their place keeps about half
of that on dense clutter, and fitting each mover's oscillation from the
robot's own observations recovers the rest, matching the oracle.

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

| Condition | Classical: given the map | own map | own map, own pose | PPO (privileged) |
|---|---|---|---|---|
| nominal | **1.000** / 0.985 | 0.970 / 0.951 | 0.850 / 0.823 | 0.960 / 0.910 |
| sparse | **1.000** / 1.000 | 1.000 / 1.000 | 0.570 / 0.569 | 0.980 / 0.963 |
| large | **1.000** / 0.990 | 0.990 / 0.977 | 0.530 / 0.524 | 0.970 / 0.940 |
| dense | **0.890** / 0.841 | 0.650 / 0.602 | 0.640 / 0.587 | 0.640 / 0.593 |
| narrow | **0.850** / 0.795 | 0.590 / 0.544 | 0.630 / 0.584 | 0.600 / 0.556 |

The learned policy is competitive in open worlds and collapses under clutter.

**Most of that clutter margin is the map, not the planner.** The second column
is the same planner building its own occupancy grid from the same 32-beam scan
the policy reads, with its pose still exact (§9.2). In open worlds it keeps
almost everything: 0.970 to 1.000. In clutter the margin goes: 0.650
against the policy's 0.640 on `dense`, and 0.590 against
0.600 on `narrow` — a tie either way at a hundred episodes, where
the mapped planner had led by 0.25. It gives up 0.240 and
0.260 to mapping, almost all of it in timeouts (0.330 and
0.390) rather than collisions (0.020), from a robot exploring
its way around obstacles it cannot see through.

**Taking its pose as well costs nothing more in clutter and most of what is
left in open worlds.** The third column adds wheel odometry, drifting, corrected by
matching each scan against the map the robot is building (§9.3). On `dense` and
`narrow` that is free — 0.640 and 0.630, against 0.650 and 0.590 with the pose
handed over — and on `large` it is ruinous, 0.530 against 0.990. The two
privileges are worth opposite things, for one reason: clutter is structure, and
structure is both what a map is needed for and what a pose can be recovered
from. §9.4 asks whether that cost belongs to this implementation: Nav2 with
SLAM pays at least as much given the same 32-beam scanner, and under a fifth of
it given 360 beams. §9.5 gives this stack 360 beams too: the pose's cost all but
vanishes, and the map's does not.

A caution on reading the last column against the first three, stronger than the
one §9.2 needed. The privileged policy is given an exact goal vector, which is
exact localisation relative to the goal, and ground-truth ranges. The third
column has neither privilege. Where it falls below the policy in open worlds
that is not a like-for-like comparison, and the comparison the experiment was
built for is between the classical columns.

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
| noisy_lidar | 1.000 / 0.985 | 0.980–0.990 / 0.966–0.976 ‡ | −0.020 to −0.010 ‡ |
| dense | 0.890 / 0.841 | 0.910–0.940 / 0.881–0.901 | +0.020 to +0.050 |
| **narrow** | 0.850 / 0.795 | **0.910–0.930** / **0.878–0.898** | +0.060 to +0.080 |

‡ Re-measured with the noise delivered (§9.20). As first published this row read
0.970–0.980 / 0.953–0.964, measured on a clean scan. The corrected passes ran at
5× real time, because the unthrottled harness now abandons episodes on a clean
sensor too; a clean `nominal` pass at the same cap on the same worlds scored
1.000, so with its privileges Nav2 does not notice the noise.

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

## 8. Results 7–11: what the sensor and its representation cost

§7 established that sensor geometry is a real constraint. Five further
experiments, each pre-registered with six seeds per arm, establish *which*
property matters, what it costs to change the representation, and how much of
either survives a reward that prices a crash like a stall.

| Experiment | Comparison | Δ `narrow` success | p |
|---|---|---|---|
| Depth vs lidar | 64 col @ 90° vs 64 beams @ 360° | −0.097 | 0.019 |
| FOV sweep | 90→360° at 64 samples, 24 seeds | ρ = +0.508 | 0.013 |
| Samples at fixed FOV | 32 vs 64 @ 90°; 64 vs 128 @ 360° | +0.002, −0.003 | 1.000, 0.955 |
| Coverage at fixed resolution | 32 @ 90° vs 128 @ 360°, both 2.81°/sample | **+0.095** | **0.024** |
| Representation | RGB + CNN vs depth + MLP | **−0.218** | **0.002** |
| Representation at 1:1 | the same, collision priced like a timeout | **−0.180** | **0.002** |

### 8.1 Coverage is causal; resolution is not

A depth camera with **4× finer** angular sampling but a 90° field of view
loses to a 360° lidar on every condition. Sweeping field of view at a fixed
sample count turns that into a monotone curve.

The decisive test pins *resolution* instead and varies coverage — the two
cannot both be held, since resolution = FOV/samples. **Doubling the sample
count changes nothing in success, twice, at both ends of the range.
Quadrupling coverage at identical angular resolution produces the entire
effect.** This sharpens the earlier framing rather than confirming it: "field
of view beats angular resolution" implies a frontier where either knob buys
performance. There is no such frontier — one knob is inert *in success rate*.

The qualifier was added late. Doubling beams at 360° moves collisions +0.083
(p = 0.006) and timeouts −0.080, so the same number of episodes fail and only
*how* changes; the pre-registered null was about success and held, and the
prose had generalised past it.

The §7 audit forecast this sweep's two interior levels — the ones it was not
fitted on — to within **+0.021 and +0.001**, untrained and unseeded: with §7.1,
two independent forecasts from one measurement, and the strongest evidence here
that the mechanism is real rather than fitted afterwards.

### 8.2 The pixels are the problem, not the geometry

The RGB camera renders the *same* geometry the depth camera measures — same
90° FOV, same 64 columns, goal vector bit-identical between modes. Only the
representation changes: pixels a CNN must interpret, rather than a vector an
MLP reads directly.

It costs **0.16–0.24 success on every condition**, with every depth seed
beating every RGB seed. Giving RGB **2.7× the compute** does not close it
(−0.162, p = 0.030) — and that rejection is null on all three outcome channels,
not just on success, which is a harder result to explain away. Nor does a
reward that prices a crash like a stall (§8.3). The render is
clean — no texture, lighting or sensor noise — so this is a lower bound. The
failure is indecision rather than recklessness: on `nominal` the encoder's cost
lands in timeouts (+0.115,
p = 0.011) rather than
collisions (+0.047,
not significant).

**A prediction that failed.** I forecast −0.05 to −0.10; the effect is two to
three times that. Worth contrasting with §8.1: the forecasts that held were
derived from a measured quantity, this one was intuition in the same confident
register.

### 8.3 The reward prices perception too

Every result above was measured under the 4:1 reward, and §9.1 shows that
ratio deciding what an observation channel is worth. Re-running the 360°
resolution contrast under indifference — `collision_penalty` 5 against a
500-step timeout at 0.01/step, one field changed, six seeds per arm — asks
whether these perception findings are about perception or about pricing. The
quantity of interest is the **interaction**, over twelve index-paired per-seed
deltas:

| b128 − b64, `narrow` | at 4:1 | at 1:1 | interaction | p |
|---|---|---|---|---|
| success | -0.003 | +0.032 | +0.035 | 0.262 |
| collision | +0.083 | -0.058 | **-0.142** | **0.0043** |
| timeout | -0.080 | +0.027 | **+0.107** | **0.0087** |

**The sign reverses.** Extra beams raise collisions under 4:1 on 6 of 6 seeds
and lower them under 1:1 on 5 of 6, while success stays insensitive to both —
which is why reading success alone found nothing. When stalling is cheap,
extra resolution is spent attempting more passages and crashing on some; when
stalling costs what crashing costs, the same resolution is spent getting
through more safely. The sensor did not change; what the objective let the
policy do with it did.

Two guards. The reward change is not free — on `narrow` both arms are *worse*
in absolute success at 1:1, collisions roughly tripling as timeouts collapse,
so **§9.1's "stacking works at 1:1" was specific to moving obstacles and does
not generalise to static clutter.** And on `nominal`, where the policy already
succeeds 93% of the time, the 1:1 resolution delta is +0.000 on success and
−0.002 on collisions: the control cell that a general property of the reward
change would have moved.

**Coverage was re-priced too, and it survives.** §8.1's headline was the
exposed one: §8.4 shows its entire benefit sitting in the timeout channel, and
indifference nearly closes that channel. Repeating the contrast at 1:1 keeps
the effect — +0.080 (p = 0.032)
on `narrow`, +0.107 (p = 0.004)
on `dense`, against +0.095 and +0.088 at 4:1 — with a success interaction that
is null on both conditions. What moves is the mechanism: collisions
-0.133 (p = 0.024)
and timeouts +0.148
(p = 0.0022). **Coverage buys the same amount under
either reward; only the failure it prevents changes.** Resolution did not
survive this test and coverage did, which is the sharpest statement of the
difference between them in this report.

**The encoder cost was re-priced last. It survives, and its failure does not
move.** Twelve new runs, seed-paired with §8.2's and differing only in the
reward, put the cost of reading pixels at −0.180 on `narrow` and −0.193 on
`dense` at 1:1, every depth seed still above every RGB seed (p = 0.002 on
both), against −0.218 and −0.238 at 4:1. The success interactions are +0.038
and +0.045 (p = 0.4697 on both). The prediction registered before training
said the cost would survive and relocate into collisions, as coverage's had;
the first half held and the second did not. Indifference collapses the depth
arm's `narrow` timeouts to 0.028, and the RGB arm still times out on
0.182 of episodes: its cost is +0.153 in timeouts (p = 0.017,
positive on 6 of 6 seeds) and +0.027 in collisions (p = 0.662).
**A coverage deficit crashes once stalling is priced like crashing; an encoding
deficit keeps stalling.** On `nominal` the cost roughly halves, −0.162 to
−0.088 (interaction p = 0.0195) — outside the registered decision, which
named `narrow` and `dense`, and not significant once corrected for three
conditions, so it is reported and not claimed.

### 8.4 What a perception deficit actually does

`seed_analysis.py` recorded success, SPL and collisions and discarded the
timeout rate, which is how §8.1's null came to be overstated. Re-running all
nine perception comparisons with the repaired tool reproduced every published
number exactly, to 0.00e+00, and turned up no further overstatement. It also
recovered a pattern none of them could show alone:

| Manipulation | Success | Collision | Timeout |
|---|---|---|---|
| coverage ↑ (90° → 360°, matched resolution) | +0.095 | +0.032 | **-0.127** |
| coverage ↓ (360° → 90°), `dense` | -0.130 | +0.008 | **+0.122** |
| resolution ↑ from inadequate (16 → 64), `dense` | +0.115 | **-0.087** | -0.028 |

Bold marks p < 0.05. **Under this reward, coverage deficits make the robot get
stuck; sensing deficits make it crash.** Both directions of the coverage
manipulation agree, and collisions never approach significance in either — on
the same six seeds that make the timeout shift significant, so this is a
contrast between channels rather than an argument from low power. Below
adequacy the pattern inverts: 16 beams is too few to see obstacles, and fixing
that cuts collisions with timeouts unmoved.

The qualifier matters: every row was measured at 4:1, and §8.3 shows the same
coverage deficit costing the same success through *collisions* at 1:1 — and
the encoding deficit, alone of the three, still costing it through timeouts,
for reasons §8.5 traces.

### 8.5 Why the pixel policy stalls

§8.3 left the encoding deficit as the one whose failure the reward does not
move. A training-free audit of the twelve 1:1 policies asks why, in two parts.

**The information is there.** The render's premise was that pixels carry the
geometry the depth vector does, and it is only approximately true: a surface
nearer than 1.2 m fills its image column, so distance survives there only as
eight-bit shading. Measured exactly, the widest span of distances that render
to an identical column is 0.069 m for walls, 0.091 m for boxes and
0.114 m for circles, the same below 1.2 m as above it. The encoder's cost is
not information the render threw away.

**Where it stalls.** Over 50 `narrow` worlds per policy, every step spent going
nowhere (under 0.15 m in 3 s) was classified by the true geometry. The RGB
policies spend 0.350 of their steps stalled against the depth policies'
0.076 (p = 0.022). Most stalls of *both* arms come after turning away
from the goal: 81% of RGB stall steps and 96% of depth's have an opening in
view and the goal-ward route blocked or out of sight, and in 86% and 100% of
those the goal is simply outside the field of view. What only the RGB policy
does is stop facing an open route to a goal in plain view — 19% of its
stall steps, none of the depth policy's.

**And at those stops its camera features see the route as open.** A linear
probe on each policy's final layer reads "goal route open" less well for RGB
than for depth (0.813 against 0.902 balanced accuracy, p = 0.004), and
at the RGB policy's open-route stalls it reads *blocked* 51% of the time,
against 15% while moving — which looked like a phantom obstacle. It was
not. The final layer also takes the robot's own velocity, near zero at any
stall, and the same probe on the CNN's image features alone reads blocked at
those stalls *less* often than while moving (24% against 35%).
The policy stops with the opening visible in its features; whatever holds it
there is downstream of the camera.

**The velocity input holds it there, and releasing it does not help.** A
replay registered before it ran overwrote the policy's velocity input and
nothing else. Told it is moving, a stalled RGB policy commands forward motion on
38.0% of its open-route stall steps instead of 1.3%; told it is stopped, a
moving one still drives on 98.4% of steps against 99.2%. Its own velocity does
not stop it, but once it has stopped, keeps it stopped. The depth policy's
stalls do not respond at all (0.7% to 0.6%). Released in closed loop,
whenever the robot has gone nowhere for 3 s, the RGB policy times out less on
every seed (−0.073, p = 0.031) and collides more on every seed by nearly as much
(+0.063, p = 0.031), while success moves by +0.010. At 1:1 that trade is
worth nothing, which is the finding: the latch is not what costs the RGB policy
its episodes. Released, its stalls become crashes, not successes.

**It cannot measure a gap.** A probe on the CNN's image features decodes the
angular width of the traversable gap toward the goal at R² = −0.29 --
worse than predicting the mean — where the same probe on the depth policy's
input vector reaches 0.220 (p = 0.002 over seeds). The same
features do carry the clearance half a metre ahead about as well as depth does
(0.205 against 0.264, p = 0.13), so this is not
a general blindness: the encoding keeps how far the wall ahead is and loses how
wide the way past it is, which is exactly the quantity a 0.22 m disc in a
corridor needs. Where the released episodes crash fits that: 13 of the 19
added collisions happen within a metre of where the robot had stalled, at a
median 0.51 m and 3.1 s after the release — it edges into something
beside the opening rather than driving off and failing elsewhere. That split by
distance is post hoc; the registered rule asked for a metre *and* three
seconds, which 9 of 19 meet, and returns MIXED.

The forecast registered in the audit script is not counted in §10's record: a
smoke test printed results before it was committed, and the commit says so.
Every clause of it held, and its headline — a policy blind to the detour — is
contradicted by the splits added afterwards. It is recorded as a forecast
whose clauses were too coarse to tell its story from the true one.

## 9. Results 12–13: where the map is wrong

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

**The motion cost is explained, given perfect prediction.** Four candidate
mechanisms were tested and eliminated: replanning churn (real, but a clutter
pathology that leaves the cost unchanged), planning failure (A\* never fails to
find a route), sensing (the movers are fully visible), and commitment length (a
6× sweep of the controller's rollout horizon moves the cost by 0.040, inside
noise, while moving absolute performance by 0.220). The fifth survived: every
actor treated a mover as a snapshot where it stood. Given the movers' exact
future positions, a planner that reasons in space-time and keeps a two-step
temporal margin loses 0.010 to motion on dense clutter and 0.000 on
sparse worlds, where the spatial baseline lost 0.145 and 0.140. What
looked like an irreducible cost of moving obstacles was a planner that could not
reason about time and, once it could, needed room for its own tracking error.
The route there — oracle prediction recovering 44% of the dense cost
(+0.070, p = 0.016), the initial plan and the controller's slow-down
recovering nothing further, agility ruled out, and timing buying access on dense
clutter (+0.050, p = 0.031) at the price of robustness on sparse
worlds (−0.050, p = 0.002) until the margin removes it — is in
[`dynamic_obstacles.md`](dynamic_obstacles.md).

**No real robot has those trajectories, and the learned policy never had them.**
Replacing the oracle with the simplest real estimate — constant velocity from
the robot's own noise-free observations — costs 0.065 on dense clutter
(p = 0.001, 14 episodes lost and 1 won), all of it collisions. The
motion cost becomes 0.075 on dense clutter and 0.020 on sparse worlds:
about half of the oracle's reduction survives on dense clutter and most of it on
sparse. Four changes to how the planner *uses* that estimate — a wider temporal
margin, withholding plans made with no spatial margin, replanning the moment an
observation contradicts it, capping how far it reaches — each failed to recover
the dense remainder, and the last shows the far end of a wrong line is worth
having: cut to 1 s, the estimate loses 0.085 on sparse worlds
(p = 0.0005). None of them changed the model, a straight line drawn along a
sinusoid. Fitting each mover's oscillation instead — three unknowns of least
squares over the track the robot has watched, told nothing about any mover —
recovers it all: +0.070 against the line on dense clutter
(p = 0.0001, 14 episodes won and none lost), and bounded-inert against
the *oracle*, leaving a motion cost of 0.005. None of the cost belonged
to the planner; each of those four was a way of coping with a wrong estimate.
The observations are noise-free and the fitted model is the one the simulator
integrates, so this is an upper bound — and a narrow one. A centimetre of
error on each observed position costs the fitted planner 0.160 on dense
clutter and the straight line 0.095, both larger than the 0.070 the model
was worth, and at that noise the two estimators are indistinguishable
(+0.005, interval [−0.050, +0.055]). The fitted model becomes the better
one again only at coarse noise, where its bounded orbit beats a diverging
line (+0.115 at 5 cm, p = 0.0002). What the estimator needed was not a
better model class but one that does not amplify its own error: fitting the
same oscillation to the observations by least squares, without differencing
them, is worth +0.145 on dense clutter at a centimetre of noise
(p < 0.0001) and is no longer distinguishable from the oracle
(−0.015, p = 0.45). Taking sight away as well — observing a mover only
when a sensor could see it — costs a 360-degree scanner −0.035 on dense
clutter (p = 0.09) and a 90-degree camera −0.125 (p < 0.0001),
leaving the camera-limited stack no better than a straight line that saw
everything. §8.1's coverage finding, from the classical side.

**Recurrence is worse, and not because of motion.** An LSTM policy — the last
untested way of supplying motion information, and the only one that learns what
to retain — is significantly *worse* than a memoryless one on fast movers
(−0.068, p = 0.017) and worse by slightly more on the slow
control (−0.078, p = 0.004). The control moving as much as the
treatment refuses the motion reading a fourth time: the cost is a general
property of the arm. Two artefacts were closed before reading it — the arm is
converged, and clearing its recurrent state costs +0.315
success, so the memory is genuinely in use. See
[`dynamic_obstacles.md`](dynamic_obstacles.md).

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

### 9.2 What the map was worth

Every classical number above is the planner reading a perfect static map. The
report has called that privilege deliberate throughout — a baseline that loses
through handicap proves nothing — and this is what it was worth. The planner
builds an occupancy grid from its own range returns as it drives, plans on it
with unknown space treated as free, and replans when something newly mapped
cuts across its route. Its pose is still exact: taking both privileges at once
would leave a result nobody could attribute.

| success, 100 worlds | given the map | own map, 32-beam scanner | own map, 90° camera | PPO |
|---|---|---|---|---|
| sparse | 1.000 | 1.000 | 1.000 | 0.980 |
| large | 1.000 | 0.990 | 0.990 | 0.970 |
| nominal | 1.000 | 0.970 | 0.980 | 0.960 |
| noisy_lidar | 1.000 | 0.940 | 0.930 | 0.960 |
| dense | 0.890 | 0.650 | 0.690 | 0.640 |
| narrow | 0.850 | 0.590 | 0.610 | 0.600 |

**Open worlds barely notice; clutter is where the map was doing the work.** The
losses are −0.240 on `dense` and −0.260 on `narrow`
(both p < 0.0001), against at most −0.030 on the three open
conditions. What the planner loses is not safety — collisions stay at
0.020 — but arrival: timeouts rise to 0.330 and
0.390 as it explores its way around obstacles it cannot see
through, replanning about 75 times an episode and finishing with half the world
still unmapped.

**And `noisy_lidar` is no longer a no-op for the classical stack.** Reading the
sensor at all means reading the condition's 0.10 m of range noise, which costs
it 0.060 (p = 0.031) where a perfect map cost it nothing.

Two cautions on reading the middle column against the policy. The comparison is
descriptive: the published PPO column is one training run's aggregate, without
per-episode outcomes to pair against, and a 0.010 difference at a hundred
episodes is nothing either way. And what is measured is *this* mapping stack --
an occupancy grid, A\*, pure pursuit and the standard recoveries. A production
stack would plausibly do better, and the project already has a Nav2 bridge to
ask with.

### 9.3 What the exact pose was worth

§9.2 took the map away and kept the pose, saying why: taking both at once would
leave a result nobody could attribute. This takes the second one, which is the
last privilege the planner holds and the one a real robot lacks most obviously.

The pose now comes from the wheels. The odometry model carries both errors a
real base has — the random walk of the standard motion model, growing with the
square root of the distance driven, and a scale and heading bias drawn once per
robot, growing with the distance itself. The second is what matters over a long
run, and is why calibration procedures like UMBmark exist. Two arms read that
estimate: one drives on it as it comes, and one corrects it by matching each
scan against the map the robot is already building — a correlative matcher over
a likelihood field, searched coarse to fine, charged for departing from the
odometry prediction, which is the front end of a standard 2-D SLAM stack.
Nothing else changes: same planner, same controller, same margins, same map,
same sensor.

The true pose still reaches the sensor, because the scanner is bolted to the
robot and not to its belief, and the diagnostics that score the estimate. It
reaches neither the map, nor the planner, nor the controller, and a unit test
holds the control path to that. Arrival is judged where the robot physically
is: a robot that believes it has arrived and has not, fails.

| success, 100 worlds | given the map | own map | own map, odometry | own map, odometry + matching | PPO |
|---|---|---|---|---|---|
| sparse | 1.000 | 1.000 | 0.670 | 0.570 | 0.980 |
| large | 1.000 | 0.990 | 0.300 | 0.530 | 0.970 |
| nominal | 1.000 | 0.970 | 0.700 | 0.850 | 0.960 |
| noisy_lidar | 1.000 | 0.940 | 0.640 | 0.780 | 0.960 |
| dense | 0.890 | 0.650 | 0.410 | 0.640 | 0.640 |
| narrow | 0.850 | 0.590 | 0.420 | 0.630 | 0.600 |

**Dead reckoning alone is not enough anywhere.** Against the same planner
holding its exact pose it is HARMS on all six conditions, from −0.170 on
`narrow` to −0.690 on `large`, every one at p < 0.001. The size tracks the
distance driven, which is what a drift model predicts: `large` has the longest
journeys and loses seven tenths of its success, `narrow` the shortest and loses
least. Collisions stay at or below 0.030. The robot does not hit things, it
fails to arrive, and timeouts reach 0.680.

**Scan matching pays most of it back, and in clutter the pose then costs
nothing measurable.** Correcting against its own map is MATTERS on five of the
six conditions: +0.230 on `dense`, +0.230 on `large`, +0.210 on `narrow`,
+0.150 on `nominal`, +0.140 on `noisy_lidar`. In clutter that closes the gap
entirely — 0.640 on `dense` against 0.650 with the pose handed over
(−0.010), and 0.630 on `narrow` against 0.590 (+0.040), neither
resolvable at a hundred episodes. The mechanism is visible in the estimate
itself: median final pose error 0.084 m on `dense` and 0.083 m on
`narrow`, against 0.329 m and 0.393 m without correction, and under a degree of
heading error against about five.

**This is the exact inverse of what the map was worth.** §9.2 measured the map
costing −0.240 and −0.260 in clutter and nothing in open worlds. The pose costs
nothing in clutter and −0.460 on `large`, −0.430 on `sparse`. One fact explains
both: clutter is structure, and structure is simultaneously what a map is needed
for — you cannot see through it — and what a pose can be recovered from — you
have something to match against. An open room is the opposite on both counts.

**Which is why matching can make things worse.** On `sparse` it does: 0.570
against dead reckoning's 0.670. The mechanism was named in the registration
before the run and is measured directly by
[`localisation_diagnostic.py`](../scripts/localisation_diagnostic.py), which
records the matcher's own score against the true error for every control period
of a val run. On `sparse` it scores a median 0.925 while the pose is 0.570 m
out; on `dense`, where it works, it scores 0.965 and is 0.058 m out. The score
it is maximising barely distinguishes the two, and its confidence carries almost
no information: restricting to the half of the steps it was most confident about
still leaves `sparse` 0.441 m out. The map is built at the estimate and the
estimate is matched against the map, so the two agree with each other while both
drift away from the world. This stack has no back end — no pose graph, no loop
closure, nothing that reconsiders a past decision — and a front end alone cannot
tell a consistent error from a correct pose.

**The sensor ordering reverses.** §9.2 found the 90° camera's forward resolution
beating the 360° scanner in clutter. With a pose to estimate as well, `narrow`
flips — 0.530 for the camera against 0.630 for the scanner (−0.100,
p = 0.0063) — while `large` goes the other way, 0.670 against 0.530
(+0.140, p = 0.038). Coverage is what pins a pose between two walls a metre
apart; forward resolution is what maps a far wall across a large room.

**On the PPO column, a caution stronger than §9.2's.** The privileged policy
reads an exact goal vector, which is exact localisation relative to the goal,
and ground-truth ranges. The two right-hand classical columns hold neither
privilege. Where the planner now falls below the policy in open worlds — 0.570
against 0.980 on `sparse` — that is a planner with no map and no pose against a
policy that still has a pose, and it is not a like-for-like comparison.

**Controls.** The full-map arm reproduces the published classical row on all six
conditions. The own-map arm reproduces Phase 6e's per-episode outcomes on all
six. And the same localised agent with its odometry noise switched off is
bit-identical to it, because the estimator is then the robot's own integrator
driven by exact wheel readings — so a difference between these arms is the
pose, not the plumbing.

### 9.4 How much of that was the implementation

§9.2 and §9.3 ended on the same caution: what they measured was *this* stack —
an occupancy grid, A\*, pure pursuit and a scan-matching front end with no back
end — and a production stack would say how much of it was the implementation
and how much the problem. This asks one. Nav2 runs on the same worlds with
neither privilege: slam_toolbox builds the map from the scan and localises
against it, with its pose graph and loop closure, and the odometry feeding it
is §9.3's own model — imported by the bridge rather than reimplemented, with
the same parameters and the same per-world seeding. It runs twice: at 360
beams, as every published Nav2 row in this report has, and at 32, the scanner
the hand-written stack built its map from.

**What is compared is what each stack loses**, not what each one scores. With
both privileges Nav2 already beats the hand-written controller in tight
corridors (§4.1), and a direct comparison without privileges would count that
head start as an answer about privileges. So each stack is scored with its
privileges and without them on the same worlds, and the statistic is the
difference in costs — Nav2's loss minus the hand-written stack's — world by
world, pooled over all six conditions. Nav2 is not deterministic, so its
with-privileges side is §4.1's two published passes and a verdict has to hold
against both.

| success, 100 worlds | hand-written, with / without | Nav2, with (2 passes) | Nav2 + SLAM, 360 beams | Nav2 + SLAM, 32 beams |
|---|---|---|---|---|
| sparse | 1.000 / 0.570 | 0.990 / 0.990 | 0.960 | 0.690 |
| large | 1.000 / 0.530 | 0.990 / 0.990 | 0.960 | 0.400 |
| nominal | 1.000 / 0.850 | 0.980 / 0.970 | 0.970 | 0.800 |
| noisy_lidar | 1.000 / 0.780 | 0.980 / 0.990 ‡ | 0.950 ‡ | 0.640 ‡ |
| dense | 0.890 / 0.640 | 0.940 / 0.910 | 0.820 | 0.620 |
| narrow | 0.850 / 0.630 | 0.930 / 0.910 | 0.810 | 0.450 |

‡ Re-measured with the noise delivered (§9.20); as first published, on a clean
scan, 0.970 / 0.980, 1.000 and 0.750. The pooled figures below use the corrected
row, and §9.20 gives the published ones beside them.

**Given a dense scanner, a production stack keeps almost everything.** Pooled
over the six conditions, the hand-written stack loses 0.290 to losing the map
and the pose; Nav2 with slam_toolbox at 360 beams loses 0.057 against the first
pass and 0.048 against the second. The difference in costs is +0.233
[+0.193, +0.273] and +0.242 [+0.202, +0.282] — IMPLEMENTATION, the registered
secondary decision. In open worlds Nav2 barely notices (0.950 to 0.970);
in clutter it pays 0.09 to 0.12, and pays it in timeouts (0.140 on `dense`,
0.180 on `narrow`) rather than collisions (0.040, 0.010), with a median pose
error under 8 cm. It is the map that costs it in clutter, not the pose — §9.2's
finding, in a different stack.

**Given the same 32-beam scanner, it pays at least as much.** Nav2 loses 0.368
and 0.360; the difference in costs is −0.078 [−0.130, −0.027] and −0.070
[−0.120, −0.018]. The registered primary decision is UNRESOLVED: the intervals
reach past the ±0.10 band a bounded null needed. What they exclude is Nav2
paying the same or less — both lie wholly below zero — so a pose graph, loop
closure and a production controller do not buy back what a sparse scanner
costs; they cost a little more, though not by the 0.10 the rule calls
material. Per condition the two
stacks fail in different places: Nav2 does better on `sparse` (0.690 against
0.570) and markedly worse in tight corridors (0.450 against 0.630, −0.180,
p = 0.0039), where its costmap, built from 32 beams, both collides (0.180) and
stalls (0.370).

**So, from Nav2 alone, what the privileges stood in for looked like the
sensor.** Holding the sensor fixed and changing the implementation to a
production one buys nothing; holding the implementation fixed and giving it a
dense sensor cuts its cost from 0.36 to 0.05. That rested on one cell inferred
rather than measured — this project's own stack at 360 beams — and §9.5 ran
it: the sensor explains the pose, and not the map.

**Getting Nav2 with SLAM to measure SLAM** took the `val` band and found seven
faults, all this project's: four in the harness, three in how Nav2 and
slam_toolbox were configured. Every one produced a plausible-looking result.
slam_toolbox 2.8 is a lifecycle node and, launched as a plain one, never
configured. An unthrottled startup loop ran the simulator at 190× real time and
starved the machine. Nav2's velocity commands queued behind SLAM's TF traffic in
a shared executor, at 0.44 per control step instead of 1.0. The global costmap,
sized to the partial map, put unseen goals outside itself. Parallel runs raced
at startup, and the cleanup script never stopped `controller_server`, because
Linux truncates process names to fifteen characters. And the one that mattered:
slam_toolbox dragged the pose *backwards* — 2.64 m of believed travel for
5.96 m driven, with perfect odometry and a scan checked to land every return on
a real surface. A probe running slam_toolbox alone in a straight line found it
in four runs: this project's own scan thresholds, scaled down "for these small
worlds" on the belief that more scans could only help. Upstream's defaults track
the same drive to 0.07 m, and the configuration now differs from them in four
documented settings. Every val number the prediction was built from was
collected after the last fix.

### 9.5 The missing cell

§9.4's conclusion rested on Nav2 alone: the hand-written stack had never been
run at 360 beams. Here it is, with nothing else changed — the scan matcher's
prior weights were tuned on the val band at 32 beams and are deliberately not
re-tuned, as slam_toolbox's were not between 32 and 360. The 32-beam arms were
run again alongside, and reproduce Phase 6f episode for episode.

| success, 100 worlds | given the map | own map, 32 | + own pose, 32 | own map, 360 | + own pose, 360 | Nav2 + SLAM, 360 |
|---|---|---|---|---|---|---|
| sparse | 1.000 | 1.000 | 0.570 | 1.000 | 0.970 | 0.960 |
| large | 1.000 | 0.990 | 0.530 | 0.990 | 0.900 | 0.960 |
| nominal | 1.000 | 0.970 | 0.850 | 0.970 | 0.960 | 0.970 |
| noisy_lidar | 1.000 | 0.940 | 0.780 | 0.830 | 0.750 | 0.950 ‡ |
| dense | 0.890 | 0.650 | 0.640 | 0.640 | 0.620 | 0.820 |
| narrow | 0.850 | 0.590 | 0.630 | 0.590 | 0.590 | 0.810 |

**The dense scanner halves this stack's cost, and all of the gain is the
pose.** Losing both privileges costs it 0.290 at 32 beams and 0.158 at 360
(+0.132 [+0.097, +0.167]). The pose's share falls from 0.190 to 0.038: the open
worlds that §9.3 found ruinous recover — `sparse` 0.570 to 0.970, `large` 0.530
to 0.900 — and median pose error falls from 0.38 and 0.44 m to 0.08 and 0.11 m.
The front end with no back end that failed there needed returns, not a pose
graph.

**The map's share does not fall at all**: 0.100 at 32 beams, 0.120 at 360. In
clutter the stack stays where it was (0.620 and 0.590, against 0.640 and 0.630),
and under sensor noise its map gets *worse* with more beams — 0.940 to 0.830 on
`noisy_lidar` — because ten times the noisy returns means ten times the evidence
for surfaces that are not there, on log-odds weights set for a sparse scanner
(§9.2).

**Against Nav2 with the same scanner, the implementation does matter.** The
difference in costs is +0.102 [+0.065, +0.137] and +0.110 [+0.073, +0.147]
against the two passes — IMPLEMENTATION, the registered decision, and only just:
the rule asks for +0.100. Very little of it is localisation: in open,
noise-free worlds the two stacks lose nearly the same (0.000 on `sparse`, +0.03
to +0.04 on `nominal`, +0.07 on `large`). It is mapping: under noise this stack
loses 0.21 to 0.22 more than Nav2 does, and in clutter 0.14 to 0.18 more, and
those three conditions carry 83% to 84% of the residual. (As first published,
with Nav2 on a clean scan, the difference was +0.112 and +0.120 and the noise
residual 0.27 to 0.28; §9.20.)

**So §9.4's conclusion was half right.** What the pose stood in for was the
sensor: given a dense one, the median final pose error is 0.07 to 0.22 m for
this stack and 0.07 to 0.23 m for Nav2 — the top of both ranges is `noisy_lidar`
— and the pose costs this stack 0.038. What the map stood in for was the
implementation: given the same dense scanner, this stack loses 0.14 to 0.22 more
than Nav2 with slam_toolbox wherever there is clutter or noise to map.

### 9.6 Repairing the dense-scan map

§9.5 left this stack's mapper as what it pays at 360 beams, and
[`dense_map_diagnostic.py`](../scripts/dense_map_diagnostic.py) says where.
Measuring the map against the world it was built from, on the `val` band: under
sensor noise, ten times the returns triple the phantom cells — occupied cells
with no real surface inside them — from 104 to 370, block 46% of the arena where
the truth is 38%, and lose 28% of the free floor to inflation around things that
are not there. In clutter it finds no phantoms at all at either beam count, and
a map that blocks *less* floor than the truth because unexplored space counts as
free. The noise regression is the mapper's; the clutter gap is not.

The obvious lever is not there either. Hits and misses are already one vote per
cell per scan, so the evidence *rate* never depended on the beam count. What a
dense scan changes is which cells get marked, and the repair follows from that:
a cell's hit is weighted by the returns it received over the returns a surface
there would have produced — its width over the arc a beam separation subtends
at its range. A lone return among six crossing beams earns a sixth of a hit.
The rule is inert wherever beams are coarser than cells, so at 32 beams the map
comes out identical and every published result stands.

| success, 100 worlds, 360 beams | own map | own map, repaired | own map and pose | own map and pose, repaired |
|---|---|---|---|---|
| sparse | 1.000 | 1.000 | 0.970 | 0.990 |
| large | 0.990 | 0.990 | 0.900 | 0.890 |
| nominal | 0.970 | 0.960 | 0.960 | 0.960 |
| noisy_lidar | 0.830 | 0.910 | 0.750 | 0.890 |
| dense | 0.640 | 0.620 | 0.620 | 0.590 |
| narrow | 0.590 | 0.590 | 0.590 | 0.600 |

**It repairs the noise regression and touches nothing else.** With its own pose
as well — the configuration the stack actually runs in — `noisy_lidar` goes
0.750 to 0.890 (+0.140, p = 0.0005, 15 episodes won and 1 lost). Every other
condition moves by at most 0.03, collisions stay at or below 0.020, and pooled
over the six the gain is +0.022 [+0.002, +0.042].

**The registered endpoint was the other arm, and it fell just short.** The
decision was placed on the own-map arm, which isolates the map by keeping the
pose exact: there `noisy_lidar` goes 0.830 to 0.910, +0.080 at p = 0.0574, 11
episodes won and 3 lost. By the registered rule that is UNRESOLVED, not
REPAIRED, and the prediction said REPAIRED. The cleaner isolation was the
weaker test: keeping the pose exact removes the compounding that makes the
repair worth twice as much when the robot has to localise on the map it is
building.

**What it is worth against Nav2.** §9.5's difference in costs at 360 beams was
+0.102 [+0.065, +0.137] and +0.110 [+0.073, +0.147] — IMPLEMENTATION. With the
repaired mapper it is +0.080 [+0.047, +0.113] and +0.088 [+0.057, +0.122]:
UNRESOLVED, no longer materially above the 0.10 band, though still above zero.
One rule in the mapper closed about a fifth of the gap a production stack had
opened. (As first published, against a Nav2 on a clean scan: +0.112 and +0.120
before the repair, +0.090 [+0.058, +0.123] and +0.098 [+0.065, +0.133] after,
the same verdicts; §9.20.)

**The mechanism is not the one the rule was designed around**, which the
registration said in advance. On val the phantom cells barely move — 370 to 309
of about 1100 occupied, and the free floor lost 0.285 to 0.270 — while success
goes 0.83 to 0.92 and replans 132 to 91. Making a phantom take several scans to
appear stops most of them from ever blocking a plan; it does not stop them
existing by the end of the episode. The median pose error under noise does not
improve either (0.217 m to 0.238 m). What improves is how often the planner is
made to change its mind.

### 9.7 Clutter: not exploring, arguing

After §9.6 the larger half of what this stack still pays at 360 beams is
clutter, and the dense-map diagnostic had already ruled out the map being wrong:
in clutter it holds no phantom cells at either beam count and blocks *less*
floor than the truth, because unexplored space counts as free. The map is
accurate and unfinished, so
[`clutter_diagnostic.py`](../scripts/clutter_diagnostic.py) measures the driving
instead, on the val band, splitting each condition by outcome.

| val, 360 beams, 12 worlds | wandering | driven / shortest | replans | reversals | spread |
|---|---|---|---|---|---|
| `dense`, successes | 0.19 | 1.15 | 38 | 1 | 2.11 m |
| `dense`, failures | 0.78 | 1.91 | 151 | 46 | 1.62 m |
| `narrow`, successes | 0.05 | 0.99 | 13 | 2 | 2.38 m |
| `narrow`, failures | 0.79 | 1.68 | 105 | 17 | 2.76 m |
| `nominal`, all succeed | 0.05 | 0.99 | 16 | 3 | 2.35 m |

*Wandering* is the share of the driving that did not end up moving the robot;
*reversals* count the steps whose heading is more than 90° from where it pointed
a second earlier; *spread* is the radius of gyration of the positions visited.

**The failures are not stuck and they are not exploring.** They spend all 500
steps, drive 1.9 times the geodesic they could have taken, and 78% of that
driving goes nowhere. They are not in recovery — the rotate-on-the-spot
behaviour fires zero times, and no plan fails at any radius. What separates them
from the successes of the same condition is reversal: 46 against 1 on `dense`.
And they cover *less* ground than the successes while driving nearly twice as
far, 1.62 m of spread against 2.11. A robot that explores covers ground. This
one changes its mind: the plan is rebuilt 151 times in 500 steps, about once
every three, and each rebuild can send it back the way it came.

**A commitment rule was tried, and the val band rejected it.** If the robot
stands by the near part of its route — adopting a new one only when the
committed one is blocked within two metres, or the new one is more than 15%
shorter — then `dense` goes 0.750 to 0.667 and `narrow` does not move. It is not
carried to the test worlds. The refusal counts say why: of about fifty replans
an episode it refuses four to six. Almost every rebuild is triggered by
something blocking the route *within* two metres, not far ahead, so a rule about
distant blockages almost never binds. The indecision is a near-field argument,
which the spread figure had already implied — on the two worst `dense` episodes
the robot never leaves a patch about a metre across while turning around 58
times and replanning every two and a half steps.

**Commitment in time was tried next, and rejected too.** If the gate is time
rather than distance — no new route within a second of the last one unless the
committed route is blocked within 0.6 m, the distance at which a blockage
matters now — then it binds as intended, refusing 34 to 49 of the rebuilds in an
episode. On forty val worlds per condition it does not help: `dense` goes 0.725
to 0.650 and `narrow` does not move from 0.600, while the symptom barely
changes (wandering 0.27 to 0.28 on `dense`, 0.34 to 0.33 on `narrow`). Twelve
worlds had made both rules look like a coin flip in either direction; forty say
one is neutral and the other costs.

**So the reversals are a symptom, not the cause.** They separate failures from
successes cleanly — 46 against 1 — and suppressing them changes nothing about
arriving. What that leaves is the map's optimism: unknown space is free, so the
planner keeps proposing routes that end at something not yet seen, and a robot
made to stand by such a route only discovers later that it does not work. The
lever is what the planner does with unexplored space, not how often it is
allowed to change its mind. Both rules stay in the code, off by default and
tested, so the comparison is reproducible;
[`clutter_commitment_val40.json`](../results/clutter_commitment_val40.json)
holds the forty-world run. **§9.8 tests that lever and it does not hold either.**

**And one failure in clutter is not planning at all.** On one `narrow` world the
robot stopped 0.49 m from a goal whose tolerance is 0.35 m, with a final pose
error of 0.50 m: it believed it had arrived. That is §9.3's failure mode
surviving into a condition where it is rare, and it is worth separating from the
rest before attributing the whole clutter gap to one cause.

### 9.8 The optimism was not the lever either

§9.7 ended by naming what was left: unknown space counts as free, so the planner
proposes routes that end at something nobody has looked at. That is a testable
claim about the cause, so it was tested.
[`src/vision_nav/planning/frontier.py`](../src/vision_nav/planning/frontier.py) plans through
**known free space only**. One Dijkstra sweep answers both questions at once —
whether the goal can be reached without entering the unknown, and what every
opening costs to reach. When it cannot, the route goes to an unseen cell chosen
by how much closer to the goal it is, with the cost of reaching it counting for a
quarter as much. Drive there, the map grows, decide again. The prediction was
registered in [`clutter_diagnostic.py`](../scripts/clutter_diagnostic.py) and
committed before the arm had been run on any seed: clutter success up by at
least +0.05, replans down by at least a quarter, and no more than 0.03 of success
given up on `nominal`, where the unknown mostly *is* free and the optimistic
shortcut is mostly right.

| val, 360 beams, 25 worlds each | as published | frontier | Δ | McNemar *p* | 95% CI |
|---|---|---|---|---|---|
| `dense` | 0.68 | 0.64 | −0.040 | 1.000 (0 won / 1 lost) | [−0.12, +0.00] |
| `narrow` | 0.60 | 0.60 | +0.000 | 1.000 (1 won / 1 lost) | [−0.12, +0.12] |
| **clutter pooled, 50 worlds** | **0.64** | **0.62** | **−0.020** | **1.000 (1 / 2)** | **[−0.08, +0.04]** |
| `nominal` | 0.96 | 0.96 | +0.000 | 1.000 (0 won / 0 lost) | [+0.00, +0.00] |

**The registered endpoint failed, and it failed by not happening at all.** Across
fifty clutter worlds the two arms disagree on three episodes — one won, two lost.
That is not a small effect measured precisely; it is a treatment that changes the
route the robot drives and leaves the outcome where it was. The pooled CI,
[−0.08, +0.04], is too wide to call a bounded null by this report's own rule, so
the honest verdict is inconclusive on magnitude and clear on the endpoint: no.
`nominal` is the one cell that is properly inert — identical outcomes on all
twenty-five worlds, CI [0.00, 0.00] — so the cost registered as a bound was the
one prediction of the three that held, and it held at exactly zero.

**The mechanism moved as predicted, and it bought nothing.** On the episodes both
arms solve, where both are doing the same job, the robot rebuilds its plan
noticeably less and steers by a shorter stretch of route ahead of it:

| val, episodes both arms solve | replans | driven / shortest | reversals | route ahead |
|---|---|---|---|---|
| `dense`, 16 worlds | 25 → 21 | 1.09 → 1.07 | 2.4 → 2.2 | 4.34 → 3.21 m |
| `narrow`, 14 worlds | 20 → 14 | 0.98 → 0.99 | 2.0 → 1.9 | 4.36 → 3.45 m |
| `nominal`, 24 worlds | 22 → 20 | 1.03 → 1.04 | 1.8 → 1.9 | 4.44 → 3.38 m |

Pooled over clutter the replan cut is 21.9%, CI [−1.8%, +40.2%] — the direction
registered, the quarter missed, and an interval that includes no change. So the
second endpoint failed too, and it deserved to: **`replans` does not mean the same
thing in the two arms**, because a frontier route ends one cell into the unseen
and is rebuilt each time the robot consumes one. Registering a threshold on a
metric that the treatment itself redefines was a mistake in the pre-registration,
and the restriction to episodes both arms solve is the repair, stated here rather
than presented as the original plan.

**And not one of these failures is a crash.** Every failing episode in this run —
39 of them, across both arms and all three conditions — spends the entire
500-step budget without a single collision. In clutter they complete 0.40 to 0.46
of the journey on average and as much as 0.96 in individual worlds. Whatever is stopping
these episodes, it stops a robot that is still driving and still upright, and
sometimes almost arrived. That makes the step budget a variable in its own right,
and it is the one quantity in this comparison that has never been varied.

**What this rules out.** Three repairs have now been tried on the same failure
and all three were rejected on val: commitment in distance, commitment in time,
and refusing to plan through the unknown. The first two suppressed the symptom
§9.7 identified without changing the outcome; the third changes what the planner
proposes — on `dense`, 64 of 72 plans go to a frontier rather than a goal, so the
treatment is unambiguously in force — and also does not change the outcome. The
reading in §9.7's last paragraph, that the map's optimism is what proposes routes
that do not work, is not supported by the test of it. Something else is keeping
these episodes from arriving, and nothing measured so far identifies it.

**A bug worth recording, because the first reading of this arm was wrong.** On
four val worlds the arm scored 0.25 against 0.75, with replans up rather than
down. The instrumentation added to find out why said it in one field: 91 of 92
plans went to a frontier and 1 to the goal. The agent was handing the planner the
*reachable* goal — the goal cell as seen through a grid where unknown counts as
blocked — which is `None` precisely whenever the floor at the goal has not been
looked at, which in clutter is nearly every plan. With no goal to score against,
the frontier choice fell back to its cost term alone: nearest unseen cell wins.
The arm was undirected exploration, not goal-directed planning, and it was
measuring a different algorithm from the one registered. One parameter had been
doing two jobs — where to route, and which way to bias the choice of opening —
and they are now separate. The planner's own tests all passed throughout, because
every one of them passes a goal explicitly; the regression test for this lives at
the agent level, and reverting the one-line call makes it fail, with the chosen
frontier closing an 8 m gap to the goal by 0.55 m.

The code stays, off by default, with an identity control pinning that it
reproduces A\* where nothing is unknown: no scored run plans this way.
[`frontier_analysis.py`](../scripts/frontier_analysis.py) holds the test and
[`frontier_diagnostic.json`](../results/frontier_diagnostic.json) the run.

### 9.9 The forensic: the margin the planner insists on

Three repairs rejected is enough to stop proposing them, so
[`clutter_forensic.py`](../scripts/clutter_forensic.py) measures the failure
instead. The instrument is the **true geodesic** from the robot's cell to the
goal, computed once per world on the grid the environment scores against and
read off every step. It is privileged information used for measurement only: it
never reaches the agent.

| val, 360 beams, 25 worlds each | best approach | final | gave back | stalled for | route held |
|---|---|---|---|---|---|
| `dense`, arrive | 0.09 m | 0.09 m | 0.00 m | 3% | — |
| `dense`, fail | 5.89 m | 7.68 m | 1.79 m | 58% | 100% |
| `narrow`, arrive | 0.10 m | 0.10 m | 0.00 m | 3% | — |
| `narrow`, fail | 5.22 m | 8.87 m | 3.65 m | 52% | 100% |

**The failures do not run out of time on the way in; they lose ground.** They
reach their closest approach around halfway through, then end 1.8 m (`dense`) to
3.7 m (`narrow`) further from the goal than they had already been, and spend
more than half the episode never beating that mark. Throughout, **every plan in
every failing episode is a complete route to the goal** — not a relaxed goal, not
a planning failure, 100% of the steps after the closest approach.

**It is not the map.** Counting the cells of the true remaining route that the
agent's own map blocks gives 23.8 on `dense` — until the same count is taken on
the *truth* at the same inflation, which gives 29.9. The agent plans at
`robot_radius + safety_margin` while the world's grid is inflated by the radius
alone, so the naive version of this measurement charges the planner's safety
margin to its mapping. Controlled, the map blocks *less* of the way home than
the world does (26.3 against 42.1 on `narrow`), which is §9.6's finding again
from a different direction. Genuine phantoms exist and are small: 8.8 cells on
`dense`, 8.9 on `narrow`.

**What the worlds themselves demand.** [`margin_audit.py`](../scripts/margin_audit.py)
asks a question with no agent in it — is the route this benchmark scores against
drivable at the clearance the planner insists on?

| val, 25 worlds each | no route at the full margin | none at half margin | none at the bare radius | margin detour |
|---|---|---|---|---|
| `nominal` | 0 | 0 | 0 | 1.03× |
| `dense` | **6** | 1 | 0 | 1.09× (worst 1.79×) |
| `narrow` | **5** | 3 | 0 | 1.17× (worst 1.95×) |

With a robot radius of 0.22 m and a safety margin of 0.18 m, **11 of 50 clutter
worlds admit no route at all at 0.40 m of clearance, and every one of them admits
a route at 0.22 m.** In the open this never happens once.

**Those are the worlds that fail.** Joining the two by seed
([`margin_overlap.py`](../scripts/margin_overlap.py), two-sided Fisher exact,
between worlds rather than between arms, nothing paired):

| val, clutter | margin-safe worlds | no margin-safe route | Fisher *p* |
|---|---|---|---|
| `dense` | 17 of 19 arrive (0.89) | **0 of 6** (0.00) | 0.00016 |
| `narrow` | 15 of 20 arrive (0.75) | **0 of 5** (0.00) | 0.0047 |
| pooled | 32 of 39 arrive (0.82) | **0 of 11** (0.00) | 8.5 × 10⁻⁷ |

Not one of the eleven is ever solved, and they account for 11 of the 18 failures
— 61% of them, on 22% of the worlds. The two groups then fail in the *same way*:
stalled for 55% against 54% of the episode, ground given back 2.43 m against
3.45 m, a complete route held 100% of the time in both. What differs is not the
mode but the rate.

**And the fallback does not rescue them.** The agent already tries three radii
and drops to a smaller one when planning fails at a larger. On the eleven it
plans at the full margin for 79% of its steps, and the smallest radius it ever
reaches averages 0.35 m — never the 0.22 m at which every one of those worlds is
solvable. The ladder rarely engages because planning rarely *fails*: the map is
optimistic about what it has not seen, so a route at full clearance keeps
appearing, is invalidated on contact, and is replanned 145 times an episode.
That is §9.7's indecision with a cause attached, and it explains why §9.8's
repair could not help — refusing to plan through the unknown does not lower the
clearance the planner asks for, so on these worlds it swaps one unreachable
proposal for another.

**Stated as a lead, not a result.** This was found by looking at val data, not
registered in advance, and the association it rests on has an obvious confound:
a world with no margin-safe route is by construction a world with a tight gap,
and tight gaps are harder to thread for reasons that have nothing to do with the
planner's margin — tracking error and pose error among them. The measurement
cannot separate those. What separates them is an experiment: relax the margin
when progress has stalled and see whether those eleven episodes are recovered,
registered before it is run. That is now item 1 of §12, and the seven failures
on margin-safe worlds are not covered by any of this.

### 9.10 The margin was blocking them, and removing it does not help

§9.9 named its own confound: a world with no margin-safe route is also just a
tight world. [`clearance_experiment.py`](../scripts/clearance_experiment.py)
separates them. The rule gives up the safety margin when the robot has stopped getting
closer to the goal for 50 steps and plans at the bare radius for the rest of the
episode. The signal is the robot's own — the distance from the pose it steers by
to the goal in its own map frame — so the geodesic §9.9 measured with never
reaches the control path. The threshold comes from the forensic rather than from
tuning: episodes that arrive spend 3% of their steps not improving, those that
fail spend 52–58%. All four endpoints were committed before the arm ran on any
seed.

| val, 25 worlds each | as published | relaxed | Δ | rule fired | collisions |
|---|---|---|---|---|---|
| `dense` | 0.68 | 0.64 | −0.040 | 10 of 25 | 0 → 8 |
| `narrow` | 0.60 | 0.64 | +0.040 | 12 of 25 | 0 → 3 |
| `nominal` | 0.96 | 0.92 | −0.040 | 3 of 25 | 0 → 1 |

**The endpoint failed at zero. Not one of the eleven worlds was recovered**,
against the four registered. The one `narrow` world the rule wins is cancelled by
the one `dense` world it loses, and every McNemar *p* is 1.000.

**But the clearance really was blocking them.** On those same eleven worlds the
robot stopped **6.29 m short before the rule and 4.80 m short after, closing more
than half a metre on 8 of 11** — one went from 5.64 m short to 0.54 m, against a
0.35 m tolerance. So the margin was holding those episodes back, exactly as §9.9
read it, and removing it still converts none of them into arrivals. Both halves
of that sentence are the result.

**What it cost is the stack's best property.** Across 75 published episodes in
this comparison the robot does not collide once. With the rule it collides 12
times — 8 on `dense`, 3 on `narrow`, 1 on `nominal` — against a registered bound
of 2. It also breaks worlds that worked: one `nominal` world that arrived within
0.34 m now crashes 4.09 m out, and the rule fired there at all only because the
robot paused near the goal. A 0.22 m robot threading a gap under 0.40 m wide has
no room for the 0.07–0.22 m of pose error §9.5 measured, and this is what that
looks like in outcomes.

Of the four registered claims, one held: success on the 39 margin-safe worlds is
unchanged at 0.821, so the rule is inert where it should be inert — though that
net hides one world lost and one won. The other three failed, including the
registered bound on collisions and the `nominal` control.

**Four repairs, four rejections, and the diagnosis is now narrower.** Commitment
in distance, commitment in time, refusing to plan through the unknown, and now
relaxing the clearance. The last one is the informative failure: it moves the
robot substantially closer on the worlds §9.9 identified and still cannot finish,
which says the binding constraint in a tight gap is not what the planner is
willing to propose but **whether this robot can drive a 0.18 m corridor of error
at the accuracy it localises to**. That is a question about the estimator and the
controller, not about the plan, and it is where §12 now points. The seven clutter
failures on worlds that had a margin-safe route all along are still unexplained.

### 9.11 The error budget: it holds its line and does not know where the line is

§9.10 offered a reading for why relaxing the clearance recovered nothing: a
0.22 m robot in a gap under 0.40 m has no room for the pose error §9.5 measured.
That is a claim about an error budget, so
[`gap_diagnostic.py`](../scripts/gap_diagnostic.py) measures the three terms in
the same units, every step:

- **room** — `clearance(true position) − robot_radius`, the lateral error the
  robot can absorb before touching something;
- **pose error** — `|estimate − truth|`. The robot steers by the estimate, so
  wherever that goes, the robot goes;
- **tracking error** — distance from the *estimate* to the route it is
  following. This is the controller's own error, measured in the frame the
  controller works in, so it carries no localisation error at all.

| val, 25 worlds each | n | room at tightest | pose error there | tracking error there | pose > room | tracking > room |
|---|---|---|---|---|---|---|
| `dense`, as published, no margin-safe route | 6 | +0.120 m | 0.065 m | 0.039 m | 5.1% | 1.0% |
| `dense`, relaxed, no margin-safe route | 6 | **+0.017 m** | **0.125 m** | 0.021 m | 28.3% | 3.9% |
| `dense`, relaxed, collided | 8 | +0.002 m | 0.118 m | 0.011 m | 20.9% | 3.9% |
| `dense`, relaxed, arrived | 16 | +0.256 m | 0.061 m | 0.026 m | 0.9% | 0.1% |
| `narrow`, as published, no margin-safe route | 5 | +0.170 m | 0.067 m | 0.003 m | 0.0% | 0.0% |
| `narrow`, relaxed, no margin-safe route | 5 | **+0.025 m** | **0.065 m** | 0.003 m | 36.0% | 2.0% |
| `narrow`, relaxed, collided | 3 | +0.003 m | 0.065 m | 0.036 m | 18.8% | 3.7% |
| `narrow`, relaxed, arrived | 16 | +0.240 m | 0.052 m | 0.016 m | 3.4% | 0.6% |

**The controller is the smaller error in every row of this table** — 0.003 m to
0.039 m at the tightest moment, always below the pose error beside it, and over
the whole episode it exceeds the room on 2–4% of steps against the pose error's
20–36%. In the tight worlds it is not that the controller is comfortable: where
the room is 0.017 m, a tracking error of 0.021 m does not fit either. It is that
the two errors differ by a factor of three to ten, so closing the controller's
would leave the episode failing on the other one. At the collisions the
controller is within 0.011 m (`dense`) and 0.036 m (`narrow`) of its own route,
while the route itself sits 0.118 m and 0.065 m from where the robot actually
was. **The robot was close to its line, and the line was in the wrong place.**

**The pose is the problem, by close to an order of magnitude.** Where the robot
arrives it has about 0.25 m of room and is wrong about its position by 0.05–0.06
m, which is comfortable. In the tight worlds with the clearance relaxed the room
falls to 0.017 m (`dense`) and 0.025 m (`narrow`) while the pose error stays at
0.065–0.125 m — three to seven times the room it has. Its belief is wrong by more
than the gap allows on 28% of steps on `dense` and 36% on `narrow`, against 2–4%
for the controller. No controller, however good, recovers an episode in which the
robot is somewhere other than it believes by more than the space it has.

**And the published stack simply does not go in.** With the margin intact, the
tight worlds bottom out at 0.120 m and 0.170 m of room, and the budget is never
exceeded on `narrow` at all. That is §9.9's finding from the inside: the
published planner refuses the gap, so the error budget never gets tested, and
relaxing the clearance is what exposes it.

**One limitation of the comparison, stated rather than buried.** Room is a
distance to the nearest obstacle and pose error is compared to it as a magnitude,
but only the component across the corridor can push the robot into a wall. An
error pointing along the corridor is counted here as though it were lateral, so
the exceedance shares above are upper bounds. The ordering they establish —
localisation error an order of magnitude above tracking error, against a room of
0.02 m — does not depend on that, because the same convention is applied to both
errors and the controller's is the smaller one either way.

**So the repair is the pose estimator.** §12's item 3 already asks for a back
end, on the argument that the 360-beam front end localises to about 0.1 m with no
pose graph at all (§9.5). This is that argument with a target attached: 0.1 m is
fine in the open, where there is 0.25 m of room, and it is the whole failure in a
gap that leaves 0.02 m. Nothing here is registered — it is a diagnostic, and the
calibration tally is unchanged.

### 9.12 The back end: it helps everywhere, and not where it was asked to

Everything above runs a **front end** alone. §12 asked for a back end on two
grounds — the 32-beam `sparse` cell, where slam_toolbox's back end beat this
stack 0.690 to 0.570 (§9.4), and §9.11's error budget, where the pose is what
fails in a tight gap. [`posegraph.py`](../src/vision_nav/mapping/posegraph.py)
is that back end: keyframes on upstream slam_toolbox's rule and its own default
thresholds, odometry edges, loop closures matched **scan against scan** rather
than against the map (the map is what the drift has already corrupted), and
Gauss-Newton least squares on SE(2) with the start pose as the gauge. When a
solve moves the poses past a threshold the map is **rebuilt** from the stored
scans at their corrected poses, because moving the estimate and keeping the old
map leaves the front end to pull the pose straight back to the stale map. All
four endpoints were registered in
[`backend_experiment.py`](../scripts/backend_experiment.py) before the arm ran
on any val seed.

| val, 25 worlds each | success | Δ | McNemar *p* | pose error p95 | keyframes | closures |
|---|---|---|---|---|---|---|
| `sparse`, 32 beams | 0.44 → 0.44 | +0.000 | 1.000 | 0.637 → 0.636 m | 40 | 20.2 |
| `dense`, 360 | 0.68 → 0.68 | +0.000 | 1.000 | **0.133 → 0.089 m** | 31 | 13.5 |
| `narrow`, 360 | 0.60 → 0.68 | +0.080 | 0.500 (2 won / 0 lost) | 0.106 → 0.089 m | 30 | 12.2 |
| `nominal`, 360 | 0.96 → 1.00 | +0.040 | 1.000 (1 won / 0 lost) | 0.119 → 0.101 m | 19 | 1.6 |

**It does what a back end is for.** Pooled over clutter the pose error p95 falls
26%, from 0.119 m to 0.089 m, which is the registered 25% by a margin of one
point. Per-episode mean error falls 30% on `dense` and the error at the moment
the episode ends falls 37%. And it is the first change tried in §9.7–§9.10 that
never costs anything: across all four cells and 100 paired episodes it wins
three and loses none, with **zero collisions in either arm** — against the
clearance rule's twelve.

**The three episodes it wins are all the same failure, and it is the right
one.** Seeds `narrow` 10009 (0.50 m → 0.34 m from the goal), `narrow` 10019
(3.05 m → 0.32 m) and `nominal` 10013 (0.58 m → 0.35 m) are all robots that had
stopped just outside a 0.35 m tolerance **believing they had arrived** — §9.3's
failure mode, which §9.7 found surviving into clutter. A better pose is exactly
the repair for it: the robot now knows it has not arrived, and keeps going.

**It does not touch the tight gaps, as registered.** None of the eleven worlds
with no margin-safe route is recovered. The pose error p95 on those worlds is
0.128 m (`dense`) and 0.082 m (`narrow`), against the 0.017–0.025 m of room
§9.11 measured — still four to six times too large. The negative prediction held,
and it held for the reason it was made: a pose graph redistributes error over a
trajectory, and 0.089 m is not 0.02 m.

**The registered endpoint failed, and the most interesting number is why.** On
`sparse` at 32 beams the back end is inert to a millimetre: 0.637 m → 0.636 m,
success unchanged. It is not idle — it accepts 20.2 closures an episode, solves
12.9 times, and moves the trajectory a total of 0.252 m. But the front end there
is already lost, at 0.472 m of mean pose error, so the closures are matched
between two poses that are both badly wrong and the constraints they contribute
are wrong with them. **A back end redistributes error; it cannot invent
information the scans do not carry.** At 32 beams on sparse worlds there is not
enough geometry in a scan to constrain a pose, and no amount of graph
optimisation supplies it. That is a sharper statement than §12's guess that the
sparse cell was where a back end would pay, and it is the opposite of it.

**And one prediction failed by helping.** `nominal` was registered to stay inside
±0.03 and came in at +0.040, reaching 1.00: the band was broken upward. It is
counted as a failed prediction because the number fell outside the registered
interval, which is what a bound means, but it is not a cost and the report says
so rather than quietly recording a miss.

Two of four registered claims held. The back end stays **off by default** — every
number elsewhere in this report is the front end alone, with an identity control
pinning that the arms agree step for step when it is off. Carrying it to the
test worlds would change the headline numbers of this report, so it needs its own
registered endpoint rather than a decision taken here; §12 says what that is.

### 9.13 The back end on the held-out worlds: smaller, and not free

§9.12 measured the back end on val — three episodes won and none lost in 100 —
and the held-out run was registered before a single held-out seed was driven
([`backend_experiment.py`](../scripts/backend_experiment.py),
`TEST_PREDICTION`), with every forecast taken from the val measurement of the
same condition and sensor. Two things were caught before launch that would have
made the run mean something else: four of the six conditions are scored on the
`test_ood` band, not `test`, and the first version drove them on the wrong
worlds; and the front end had to be shown to be the published arm, not assumed
to be. **It reproduces Phase 6i's published outcomes on 100 of 100 episodes in
each of the five 360-beam cells**, so every contrast below is against the
numbers this report publishes.

| held-out, 100 worlds each | success | Δ | McNemar *p* | pose error p95 |
|---|---|---|---|---|
| `sparse`, 32 beams | 0.58 → 0.60 | +0.020 | 0.500 (2 won / 0 lost) | 0.564 → 0.582 m |
| `dense` | 0.59 → 0.61 | +0.020 | 0.727 (5 / 3) | 0.127 → 0.097 m |
| `narrow` | 0.60 → 0.59 | −0.010 | 1.000 (3 / 4) | 0.139 → 0.115 m |
| `nominal` | 0.96 → 0.97 | +0.010 | 1.000 (1 / 0) | 0.097 → 0.089 m |
| `large` | 0.89 → 0.91 | +0.020 | 0.500 (2 / 0) | 0.173 → 0.140 m |
| `noisy_lidar` | 0.89 → 0.89 | +0.000 | 1.000 (1 / 1) | 0.254 → 0.249 m |
| **pooled, 600 paired** | **0.752 → 0.762** | **+0.010** | **0.286 (14 / 8)** | |

**The registered endpoint held, by its letter, and the effect shrank.** Pooled
over 600 paired episodes success rises 0.010, CI [−0.005, +0.025], inside the
registered [0.00, +0.05]. The registration named this outcome in advance: a gain
at the floor of the interval satisfies it while saying the val result was
largely noise, and the report was to say so. It is not zero, and it is not
distinguishable from zero. **And val's "costs nothing" does not replicate**: on
the held-out worlds the back end wins 14 episodes and loses 8, where val won
three and lost none. It is a small net benefit with real churn, not a free one.

**The pose result held, at the edge.** Clutter pose error p95 falls 20.2%, from
0.133 m to 0.106 m, against a registered 20–32% — inside by two tenths of a point.
`sparse` at 32 beams stayed inside its registered ±0.03, at +0.020; it won two
episodes on held-out where val had moved none, and its pose error got slightly
worse rather than better.

**The collision bound failed, and the reason is a fact I got wrong.** I
registered at most two collisions with the back end, on the stated ground that
"the front end has never collided in any measurement in this report". It has:
the published front end collides 4 times in these 600 episodes, and the back end
also collides 4 times — but not in the same worlds. On `dense` and `narrow` it
avoids one collision each and causes one each elsewhere; on `noisy_lidar` both
arms collide in the same two worlds. The net is zero, and a bound built on a
false premise about this report's own numbers was the wrong instrument for it.

**It does nothing under noise.** On `noisy_lidar` success is unchanged and the
pose error p95 moves 2%. Of 5.5 closure attempts an episode it refuses 2.5,
against about one in ten on `dense` and `narrow` and one in five in the open
cells. This paragraph first gave the reason as noisy scans matched against a
noisy map; that was wrong, because closures here are matched scan against scan
and never read the map, and §9.15 measures what is actually happening — closures
under noise are plentiful and correct in exactly the episodes that fail, and the
episodes fail anyway. §9.14 shows the map is not what limits `noisy_lidar`
either, which leaves the pose under noise as the one thing neither repair reaches
alone.

**What this does to the report.** Nothing to the headline numbers: they remain the
front end alone, because a pooled +0.010 that cannot be told from zero is not a
reason to replace them, and the back end stays off by default. Three of the four
registered claims held — all three forecast from a val measurement of the same
quantity — and the fourth failed on a premise about this report that was not
true, which is the calibration lesson of §10 in miniature.

### 9.14 The noisy map, repaired at its source — and it was not what limited `noisy_lidar`

§9.6 left `noisy_lidar` as this stack's worst open condition at 360 beams, 0.890
on the held-out worlds against Nav2's 0.950 (‡, with the noise delivered;
1.000 as first published), with 309 phantom cells still standing and 28% of the
free floor lost to inflation around them. §12 proposed
the next thing to try: count a miss once per *beam* that passes through a cell,
not once per cell per scan. The code's own comment says that vote exists to stop
the samples of one beam voting several times; collapsing separate beams as well
is not what a textbook inverse sensor model does.

[`clearing_diagnostic.py`](../scripts/clearing_diagnostic.py) measures that
open loop, which isolates the map. The robot drives with the published mapper,
and shadow maps are handed the identical scans at the identical poses. A null
shadow with no rule must end every episode identical to the robot's own map —
grid and evidence both — and did on all 36 episodes.

| `noisy_lidar`, 12 val worlds | phantoms | beyond 0.5 m of any surface | surface recall | floor lost |
|---|---|---|---|---|
| published mapper | 309 | 190 | 0.806 | 0.270 |
| one miss per beam | 288 | 186 | 0.779 | 0.264 |
| **obstacle range** | **21** | **0** | 0.754 | **0.051** |

**The rule §12 proposed is the wrong lever.** Per-beam clearing takes the
phantoms from 309 to 288 and costs a little recall. The diagnostic also recorded
how far each phantom sits from the nearest real surface, and that is the finding:
190 of the 309 sit more than half a metre from anything real, which range noise
of 0.1 m cannot do.

**The sensor can.** `Lidar2D.scan` adds its noise *after* clipping to maximum
range and then clips again, so a beam that hit nothing reads 6 m plus noise,
clipped — and half of those come back a little under 6 m. The mapper took every
reading below the maximum as a surface, so half the beams crossing open floor
planted a phantom at the edge of the sensor's reach. Corroboration cannot help
there, since a real surface at 6 m also returns one beam per cell, and almost no
later beam passes through those cells to clear them.

**The repair is Nav2's `obstacle_max_range`.** Readings within three standard
deviations of the sensor's own range noise of its maximum still clear the space
they cross, but never mark a surface. The margin is a datasheet figure the robot
has for its own sensor, not anything about the world. With a noise-free sensor the
line sits at the maximum, so the rule is inert there by construction: `nominal`
and `dense` came out identical open loop (456 and 514 occupied cells on both
arms), and on `nominal` closed loop every episode is identical in every field. On
`noisy_lidar` the phantoms fall 93%, every one beyond 0.2 m is gone — the 21 left
hug real walls, which is genuine range noise — and the floor lost falls to 0.051,
beside the noise-free `nominal` figure of 0.047. The cost is recorded rather than
buried: recall falls 0.052, because walls between 5.7 m and 6 m are now marked
only once the robot is closer.

**And it barely changes whether the robot arrives.** Closed loop, on all 100 val
worlds:

| `noisy_lidar`, 100 val worlds | success | McNemar *p* | collisions | replans | pose error, median |
|---|---|---|---|---|---|
| front end | 0.87 | | 2 | 61 | 0.141 m |
| with the obstacle range | 0.89 | 0.754 (6 won / 4 lost) | 0 | 59 | 0.099 m |

+0.020, CI [−0.04, +0.08]. It removes both collisions and cuts the pose error by
30%, which is the map repairing the matcher that reads it — and it moves arrivals
by two episodes in a hundred. Twenty-five worlds had shown 0.92 against 0.92; a
hundred show a small gain that is not distinguishable from none, and the first
twenty-five reproduce identically in every field inside the larger run.

**So the map is no longer what limits `noisy_lidar`.** Under noise it is now about
as clean as the noise-free map, and success hardly moves. What is still twice as
wrong is the pose — 0.099 m median under noise against 0.049 m on `nominal` —
which is §9.11's error budget again, from a different direction.

**Not carried to the held-out worlds.** Val supports no harm, fewer collisions and
a better pose; it does not support a success gain, and a held-out run whose honest
forecast is +0.02 ± 0.06 would be a registered coin flip. Nothing here was
registered, so the calibration tally is unchanged. The rule stays off by default,
with the identity controls above pinned; adopting it is a decision about the
published stack rather than a finding.

### 9.15 Both repairs together, and why correct loop closures do not rescue `noisy_lidar`

§9.13 left the back end inert under noise and §9.14 left the obstacle range
moving arrivals by two in a hundred. Whether the two together close the gap that
neither closes alone is a factorial question, so
[`combined_noisy_experiment.py`](../scripts/combined_noisy_experiment.py) runs
the 2×2 on all 100 val worlds, one arm per process. The two arms already
measured in §9.14 reproduce that run on 100 of 100 episodes each.

| `noisy_lidar`, 100 val worlds | success | collisions | pose error, median | pose error, p95 |
|---|---|---|---|---|
| front end | 0.87 | 2 | 0.141 m | 0.277 m |
| obstacle range | 0.89 | 0 | 0.099 m | 0.206 m |
| back end | 0.89 | 0 | 0.138 m | 0.263 m |
| both | 0.89 | 0 | 0.092 m | 0.193 m |

**Together they are no better than either alone.** Each rule moves success by
+0.020; the pair moves it by +0.020; the pair against the obstacle range alone is
two won and two lost, and the interaction is −0.020. The pose gains roughly stack —
the pair gives the best pose of the four — and arrivals do not move. Not carried
to the held-out worlds.

**The reason I gave for running it was wrong, twice, and the result was expected
before the data.** §9.13 and §12 said a cleaner map would help the closures,
because closures score against the map. They do not: this back end matches scan
against scan and never reads the map, so the obstacle range could not reach the
closures at all. Applying the obstacle range to the keyframe scans instead was
measured before the run and made closures *worse* — on six val worlds the share
of closure attempts clearing the score gate fell from 29% to 14% — and was
reverted. With both mechanisms gone, the expectation written down before the
run was that the pair would land where the obstacle range alone does, and it did.

**What actually fails.** With the obstacle range on, 5 of the 11 failures are a
robot stopped just outside the 0.35 m tolerance believing it has arrived, with a
median pose error of 0.47 m over the episode against about 0.09 m for those that
arrive. [`noisy_localisation_diagnostic.py`](../scripts/noisy_localisation_diagnostic.py)
traces them. **The error drifts; it does not jump**: the largest one-step rise is
0.011 m, against 0.009 m in episodes that arrive, and the stuck episodes end
0.529 m out against 0.162 m. The scan matcher returns no correction on 85% of
steps in the stuck episodes and 79% in the arrivals — close enough that it does
not separate them.

**And the back end is not missing where the robot fails.** Across the 100 val
worlds, the failing episodes accept 27.5 closures each and the arriving ones
0.8; 69 episodes form no closure at all. Failures wander and revisit, so they
generate the closures — and the closures are right. Scored against the *true*
relative pose, 299 of the 302 closures accepted on twelve noisy worlds are within
0.15 m of it and none is beyond 0.30 m. Closures under noise are plentiful and
correct in exactly the episodes that fail, and the episodes fail anyway. (The
share of attempts clearing the gate depends heavily on which worlds are drawn:
29% on the first six, which are mostly arrivals that barely revisit anything,
and 81% on twelve, which include episodes that circle and generate hundreds of
attempts between them. That is the same finding, not a contradiction of it.)

**So the noise-aware closure match was not built.** Widening the blur by the
sensor's noise on both scans would have admitted 348 of the 373 attempts with one
wrong, against 302 with none — more of something that is already present and
already not helping. It is inert on clean scans by construction, and came out
identical on `nominal`.

**Open, and stated as a hypothesis rather than a finding.** Why do correct
closures not fix the drift? The reading that fits is that they link keyframes
that drifted *together*, late in an episode spent circling where the robot
thinks the goal is: right relative to one another, and silent about the absolute
error, which only a closure back to the well-localised keyframes near the start
could measure. §12 says what would test it. Nothing here was registered, and the
calibration tally is unchanged.

### 9.16 Where the closures reach — tested through the wrong quantity

§9.15 left a question with two answers that leave different marks. Correct
closures in the failing `noisy_lidar` episodes might change nothing because they
link keyframes that drifted *together* — or because, reaching well-localised
keyframes, they are outvoted in the solve.
[`closure_reach_diagnostic.py`](../scripts/closure_reach_diagnostic.py) records,
for every accepted closure, the true error of both keyframes when each was
added, and for every solve with closures, the newest keyframe's true error before
and after. Three claims for the co-drift reading were committed before any val
world was driven, along with the signature the alternative would leave. The
worlds were fixed in advance — every val world that formed a closure in §9.15's
run — and the run reproduces that one on all 31 of them, outcome and closure
count alike.

| failing episodes, 10 worlds, 303 closures | measured | registered for co-drift | |
|---|---|---|---|
| median error of the anchor when it was added | 0.235 m | above 0.30 m | failed |
| closures reaching an anchor within 0.15 m | 37% | below 20% | failed |
| median change in the newest error per solve | 0.017 m | below 0.05 m | held |

More than a third of the closures in the failing episodes reach back to a
keyframe that was still within 0.15 m of the truth when it was added, and over
117 solves the newest keyframe's error moves by 0.017 m at the median — in the
direction of *worse*, by 0.012 m. Of three claims, one held, and the scores stand
as registered.

**The conclusion first drawn here was wrong, and §9.17 says why.** This section
read the result as refuting the co-drift hypothesis and concluded that "the graph
is being told about a real disagreement" which the solve then ignores. It is not
being told one. The claims tested co-drift through each anchor's *absolute*
error, which does not measure it: two keyframes can each be a quarter of a metre
off and agree perfectly with one another if they drifted together. What decides
it is the drift *between* the linked pair, and §9.17 measures that — 0.028 m at
the median. The well-anchored closures came from keyframes that were themselves
well localised, early in the episode, before anything drifted. The hypothesis
held in substance; the registration tested it through a proxy.

### 9.17 The closures agree with the drift: why the back end cannot see it

§9.16 left two candidate causes for a solve that did not use correct closures —
closures weighted below odometry, and closures anchored to drifted keyframes —
and they can be separated without driving an episode differently.
[`closure_resolve_diagnostic.py`](../scripts/closure_resolve_diagnostic.py)
drives the ten failing val worlds once with the published back end, saves each
graph, and re-solves it offline from the keyframes' estimates four ways. The
control comes first, and holds: the offline re-solve with the published settings
reproduces the pose each robot actually ended with, 0.000 m apart at the median.

| failing episodes, final keyframe's true error, median | error | cut |
|---|---|---|
| robot as it ended | 0.388 m | |
| front end alone, no solve | 0.388 m | |
| published re-solve | 0.388 m | |
| closures weighted as odometry | 0.388 m | −0% |
| only the closures with well-localised anchors | 0.408 m | −5% |

**By the registered rule, neither cause.** My guess was that dropping the
drifted-anchor closures would cut the error by a quarter and raising the weights
would not; the first failed — the oracle filter made it slightly *worse* — and
the second held. The rule's third branch, "something about the solve itself",
is what it returns.

**That branch was misconceived, and one row says why.** The front end with no
solve at all, the published re-solve and the robot all end at the same 0.388 m.
A Gauss–Newton solve that moves nothing is not a broken solve; it is a solve
whose measurements already agree with its estimates. The exploratory check that
follows was added after the registered test, from the same saved graphs, and is
labelled as such:

| 303 closures in the failing episodes | median |
|---|---|
| closure against the *true* relative pose | 0.048 m |
| closure against the relative pose the drifted chain implies | 0.054 m |
| drift *between* the two keyframes a closure links | 0.028 m |
| absolute error of each of those keyframes | 0.23 m |

**The closures agree with the drift.** They link keyframes whose drift relative
to one another is 0.028 m while each is 0.23 m from the truth, so they measure the
relative pose accurately, the chain already has that relative pose right, and the
disagreement the solve sees — 0.054 m — is the size of the measurement itself;
only 10% of closures disagree by more than 0.10 m. The 113 closures with a
well-localised anchor came from newest keyframes that were *also* well localised,
0.073 m out: early in the episode, before anything drifted. Not one closure spans
the drift. Re-solving moves any keyframe by 0.067 m at the median episode and
0.115 m at most, which is the solve doing the little there is to do.

**So the co-drift hypothesis held, and §9.16 tested it through a proxy.** The drift
accumulates in the front end during a stretch where the robot revisits nothing it
saw while still well localised; when it later circles, it revisits only places
seen after the drift set in, and every closure it can make links two keyframes
that drifted together. A loop-closing back end can correct drift only across a
loop that closes back over it, and in these episodes none does. That makes the
`noisy_lidar` failures unobservable to this back end by construction, and it puts
the remaining repair where the drift is created — the front end under noise — or
in a closure that reaches back across it, which this planner gives the robot no
reason to make.

Of the two registered claims one held. The proxy is its own calibration lesson,
recorded in §10: a hypothesis registered through a quantity that does not measure
it can fail its test and still be right.

### 9.18 The motion prior is not the lever either

§9.17 put the `noisy_lidar` drift in the front end, and the first candidate there
was the scan matcher's motion prior: chosen on val in §9.5 to keep about 0.03 m of
match noise from being taken seriously, it might, on a score surface flattened by
range noise, override the scan when the scan is right. Binding alone would prove
nothing — a prior that overrides a noisy argmax is doing its job — so
[`prior_binding_diagnostic.py`](../scripts/prior_binding_diagnostic.py) computes,
at every scan match, the odometry's prediction, the published correction and the
answer with the prior removed, and scores each against the ground truth. Three
claims were committed before any val world was driven, each beside the reason
its quantity measures the idea, which is the defence §10 now prescribes.

| val scans | prior binds | overridden answer closer to truth | median error: predicted / published / no prior |
|---|---|---|---|
| `nominal` | 46% | 55% | 0.039 / 0.038 / 0.036 m |
| `noisy_lidar`, all | 87% | 49% | 0.210 / 0.209 / 0.213 m |
| failing episodes | 86% | 58% | 0.318 / 0.318 / 0.331 m |
| arriving episodes | 91% | 15% | 0.123 / 0.125 / 0.147 m |

**Noise roughly doubles how often the prior binds** — 87% of scans against 46% on
`nominal`, a ratio of 1.9 against the registered 2. **And the answers it overrides
are a coin flip**: closer to the truth on 49% of the scans where it binds. So it is
not systematically suppressing real corrections, and removing it would make the
median error *worse* in both groups — 0.318 m to 0.331 m in the failures, 0.125 m
to 0.147 m in the arrivals. The scan does point the right way more often in the
failing episodes than in the arriving ones, 58% against 15%, which is the one
claim of three that held; but it is not reliable enough there to correct anything,
and it is wrong by more when it is wrong.

**One structural fact is worth more than the rest.** In the failing episodes the
odometry's prediction and the published correction are the same at the median —
0.318 m from the truth — and that is four times the matcher's search window, which
reaches ±0.08 m. Once the drift has outgrown the window, no setting of the prior
can bring the pose back in a step; the matcher can only agree with a map that the
same drift built. Under 0.1 m of range noise the match against that map is not
informative enough to stop the drift while it is still small, and neither the prior
nor the back end changes what the match can see.

**What that leaves.** Five candidate levers on `noisy_lidar` have now been measured
and set aside: clearing scaled by beam count, a cleaner map for closures that never
read it, closures matched with a noise-widened blur, closures reweighted or
filtered in the solve, and the motion prior. The pose under noise is what limits
`noisy_lidar`, as §9.11 found the pose is what limits the tight gaps, and the
levers that leave the scan matcher's search unchanged are spent. §12 says what is
left.

### 9.19 slam_toolbox's front end, rebuilt: it smooths the drift and does not remove it

§12 named a different scan matcher as the one lever left on `noisy_lidar`.
[`frontend.py`](../src/vision_nav/mapping/frontend.py) copies the three ways
slam_toolbox's matcher differs from this stack's, with the values in the repo's
own `slam_params.yaml`: a match only every 0.5 m or 0.5 rad of travel, against a
buffer of the last 10 keyframe scans rather than the whole map, over ±0.25 m and
±0.349 rad of heading — fifteen times this stack's heading window — with Karto's
deliberately weak tie-break towards the odometry. It reuses the back end's
correlative machinery, and its tests recover offsets well past the old matcher's
reach. [`frontend_experiment.py`](../scripts/frontend_experiment.py) measures it
against the published map matcher on all 100 val worlds. The expectation was
written down before that run: better pose under noise, worse on clean worlds,
success roughly unchanged.

| 100 val worlds | success | McNemar *p* | pose error, median | pose error, final |
|---|---|---|---|---|
| `noisy_lidar` | 0.87 → 0.83 | 0.52 (9 won / 13 lost) | 0.141 → 0.113 m | 0.251 → 0.252 m |
| `nominal` | 0.95 → 0.99 | 0.125 (4 / 0) | 0.047 → 0.060 m | 0.068 → 0.120 m |
| `dense` | 0.72 → 0.73 | 1.000 (4 / 3) | 0.052 → 0.079 m | 0.078 → 0.163 m |

**It does not close `noisy_lidar`; it makes it slightly worse.** Success falls by
0.040, nine episodes won against thirteen lost. The median pose error under noise
does improve, by a fifth — the expectation's first clause — and the number that
decides the episode does not move: the pose error at the end of the episode is
0.251 m before and 0.252 m after. **The buffer drifts with the robot.** Matching
against the last ten keyframes bounds how fast the drift builds, which is why the
median improves, and holds no more absolute reference than the map did, which is
why the drift that ends the episode is untouched. That concern was stated in the
design before any of this was run.

**And it costs the clean conditions their pose.** Matching only every half metre
lets the odometry drift in between where the published matcher tracks every
step, so the final pose error roughly doubles — 0.068 m to 0.120 m on `nominal`
and 0.078 m to 0.163 m on `dense`. Success there does not suffer: `nominal` moves
four episodes the right way and none the wrong way, and `dense` is level, but at
p = 0.125 the first cannot be told from noise.

**Not carried to the held-out worlds.** Val does not support it on the condition it
was built for, and it costs pose accuracy everywhere else. It stays in the code,
off by default, with an identity control pinning the published arm. Nothing was
registered, and the calibration tally is unchanged.

**Is slam_toolbox's pose even better? Measured — first on the wrong sensor.** Six
levers were aimed at localisation on the premise that localisation is why Nav2
reaches 1.000 on `noisy_lidar`. The Nav2 runs record each episode's final
believed-against-true pose error, so the premise can be checked on the same
held-out worlds, and the odometry is identical: the ROS bridge drifts its `odom`
frame with this stack's own `DeadReckoning` and default `OdometryConfig`, seeded
the same way.

| held-out `noisy_lidar`, 360 beams | success | final pose error, median | max |
|---|---|---|---|
| Nav2 + slam_toolbox, clean scan (as first published) | 1.00 | 0.075 m | 0.294 m |
| Nav2 + slam_toolbox, noise delivered (§9.20) | 0.95 | 0.225 m | 0.826 m |
| this stack | 0.89 | 0.238 m | 1.043 m |

As first published, this paragraph compared the first row with the third: on the
89 worlds where both arrived — so that this stack's 500-step failures cannot
inflate its figure — slam_toolbox ended 0.074 m from the truth and this stack
0.228 m, worse on 90% of them, and it concluded that slam_toolbox localises three
times better "from the same odometry and the same scans". The odometry was the
same. The scans were not: the bridge had delivered every `noisy_lidar` scan
without its noise (§9.20), and I had taken "the same scans" from the bridge's
configuration — the check that missed the bug — rather than from a scan it
published.

**On the same scans, slam_toolbox localises no better than this stack.** With the
noise delivered, on the 87 worlds both complete, slam_toolbox ends 0.220 m from
the truth and this stack 0.228 m; slam_toolbox is the closer on 47 and this stack
on 40 (sign test p = 0.52). Noise triples slam_toolbox's final pose error, 0.075 m
to 0.225 m. The factor of three was the sensor bug, all of it, and the front end
copied here was being measured against a target that did not exist.

**What that leaves.** The six levers of §9.14–§9.19 were aimed at a localisation
gap that a production stack does not close either. What separates the two under
noise is arrival — 0.950 against 0.890, eight worlds won and two lost
(McNemar p = 0.11), not resolved at a hundred worlds — and if it is real, it is
not localisation, because the pose error is the same. It would be in what each
stack does with a fifth of a metre of pose error: the planner and the controller,
where §9.11's tight gaps already point. slam_toolbox does link every new scan to
every earlier scan within 1.5 m (`link_scan_maximum_distance`) at a response of
0.1 (`link_match_minimum_response_fine`), where this back end links consecutive
keyframes and adds rare closures six apart behind 0.55; that is a fact about the
two configurations, and no longer the explanation of anything.

### 9.20 The noise, delivered: what Nav2 does with the sensor it was meant to have

Building the dense graph §9.19 first proposed began with checking what Nav2
receives, and Nav2 had never received the noise. The ROS bridge built its scanner
from the condition's `LidarConfig`, so `noise_std` was 0.10 m — which is what the
fairness check in [`ros2_bridge/README.md`](../ros2_bridge/README.md) looked at —
and then called it without a random generator, and `Lidar2D` adds noise only when
it is handed one. At runtime the bridge's scans were identical to a noise-free
scanner's; given a generator, the same scanner delivers the configured noise,
which is what its test now checks.
Every Nav2 run on `noisy_lidar` — §4.1's two passes and §9.4's two SLAM arms — had
been scored on a clean sensor, and since `noisy_lidar` shares `nominal`'s worlds,
each was one more pass of `nominal`.

[`bridge_sensor.py`](../ros2_bridge/bridge_sensor.py) applies the noise, seeded
from the world as this stack seeds its own sensor, and audits every scan against
the same scan cast without noise; a run whose delivered noise disagrees with its
configuration stops after its first episode. Its test checks the delivered scan
and fails on the old bridge. Every run below delivered 0.100 m. The four runs
were repeated under a registration committed first
([`nav2_noise_rerun.py`](../scripts/nav2_noise_rerun.py)), whose primary was that
slam_toolbox, on the worlds both stacks complete, ends closer to the truth by a
factor of 1.5 to 2.5 — less than the three §9.19 had reported, and BETTER.

| held-out `noisy_lidar` | success, clean scan (as published) | success, noise delivered | final pose error, median |
|---|---|---|---|
| Nav2, full privileges, 2 passes | 0.970 / 0.980 | 0.980 / 0.990 (5× real time) | — |
| Nav2 + slam_toolbox, 360 beams | 1.000 | 0.950 | 0.075 → 0.225 m |
| Nav2 + slam_toolbox, 32 beams | 0.750 | 0.640 | 0.226 → 0.400 m |

**On the same sensor, slam_toolbox localises no better than this stack.** On the
87 worlds both complete it ends 0.220 m from the truth and this stack 0.228 m, a
ratio of 0.97; slam_toolbox is the closer on 47 worlds and this stack on 40, sign
test p = 0.52. UNRESOLVED, where the registration said BETTER. Its success edge,
0.950 against 0.890, does not resolve either (8 worlds won, 2 lost,
McNemar p = 0.11).

**The full-privilege arm, as registered, measured the harness.** Its two passes
ran unthrottled, as §4.1's had, and scored 0.720 and 0.730: Nav2 abandoned 27
episodes in each without the robot moving, on worlds that mostly differed between
the passes — of the 16 and 19 episodes that never received a command, two were
the same world. Read at face value, that is the noise costing Nav2 a quarter of
its success with the true map and pose in hand. A clean control said otherwise.
On 30 val worlds, each run alone: unthrottled `noisy_lidar` 0.767 with 7
abandoned; unthrottled `nominal`, with no noise at all, 0.733 with 8;
`noisy_lidar` held to the SLAM arm's 5× real time, 1.000 with none. The
unthrottled loop now abandons episodes on a clean sensor too — the fault
`run_nav2_eval.py` documents for unthrottled runs, which can outrun the global
costmap's switch to a new world's map. §4.1's twelve published passes recorded one
episode without a command between them, so they were not touched by it. An
addendum, committed before its runs and disclosing all of this, repeated the arm
at 5× with a clean `nominal` pass at the same cap on the same worlds: 0.980 and
0.990 against 1.000, the noise worth −0.020 (McNemar p = 0.50) and −0.010
(p = 1.00). With its privileges, Nav2 does not notice the noise.

**No pooled verdict changes.** With the corrected cells — the throttled passes for
the full arm — every difference in costs moves by 0.010 or 0.020, because Nav2 now
pays something under noise (0.03 to 0.04 with SLAM at 360 beams, 0.34 to 0.35 at
32) where on a clean scan it paid nothing:

| difference in costs, two passes | as published | corrected | verdict |
|---|---|---|---|
| §9.4, 32 beams | −0.058 / −0.050 | −0.078 [−0.130, −0.027] / −0.070 [−0.120, −0.018] | UNRESOLVED |
| §9.4, 360 beams | +0.243 / +0.252 | +0.233 [+0.193, +0.273] / +0.242 [+0.202, +0.282] | IMPLEMENTATION |
| §9.5, 360 beams | +0.112 / +0.120 | +0.102 [+0.065, +0.137] / +0.110 [+0.073, +0.147] | IMPLEMENTATION |
| §9.6, repaired mapper | +0.090 / +0.098 | +0.080 [+0.047, +0.113] / +0.088 [+0.057, +0.122] | UNRESOLVED |

Under noise this stack loses 0.21 to 0.22 more than Nav2, not 0.27 to 0.28. At
32 beams both intervals now lie wholly below zero: with the same sparse scanner
Nav2 pays a little more than this stack for losing the map and the pose. §9.5's
IMPLEMENTATION now clears its +0.100 threshold by 0.002 and 0.010. Re-scoring the
registered predictions those sections made changes one clause: §9.4 had Nav2's
median final pose error at most 0.20 m on every condition at 360 beams, and on
`noisy_lidar` it is 0.225 m.

**What the registration got right and wrong.** The primary failed. The SLAM arms'
success held at both beam counts, and the 32-beam pose clause held; the 360-beam
pose clause (0.08 to 0.18 m) failed; the full-arm clause failed as registered, for
the harness's reason; and the re-scored verdicts, read off the registered passes,
moved in directions it had not predicted. The addendum held on every clause — and
it was forecast from a val measurement of the same quantity at the same cap,
which is the easiest kind of prediction to get right.

**Two lessons, one per fault.** A fairness check on a configured value is a check
on the configuration; the one that would have caught this asks what reaches the
other side, and the bridge now asks it of every scan. And the clean control is
what kept the harness's failure from being published as a finding about noise:
without the unthrottled `nominal` run, 0.72 would have read as the noise costing
Nav2 a quarter of its success with the map in hand.

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

**The reward decides what information is worth.** Section 5.3 found the 4:1
collision-to-timeout ratio makes the policy stall rather than get through. The
same ratio also sets what *extra* information buys, for a learned channel
(§9.1: frame stacking's collision saving is swallowed by timeouts at 4:1 and
becomes successes at 1:1) and for a sensor parameter alike (§8.3: doubling
angular resolution raises collisions at 4:1 and lowers them at 1:1, the
interaction significant at p = 0.004 while success moves in neither). A
channel is worth only what the objective lets the policy do with it — worth
establishing before concluding that a sensor or a representation is useless.

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

**Calibration.** Of fifty-eight predictions made in advance, fifteen held. Eight
were derived from a *measurement* of the same quantity — two to within 0.021 and
0.001, one on both magnitude and mechanism, one whose magnitude came from
measuring the estimator it was about, three of §9.13's four, each forecast
from the val result for the same condition and sensor, and §9.20's addendum,
forecast from a val run of the same arm at the same cap. Two more are weaker in
kind and are counted as held anyway: §9.8 and §9.10 each registered a *bound* on
what a treatment would cost rather than a direction, and both came in at zero. A
bound is the easiest form of prediction to satisfy, which is worth saying plainly
in a tally that otherwise counts point forecasts. Two more are §9.12's: the only
*negative* prediction in the set — that a back end would not recover the tight
gaps — which held for the arithmetic it was made from, and a pose threshold set
from a train-band check. The last three are §9.16's third claim, §9.17's second
and §9.18's third. Twenty-six from extrapolation, intuition, arithmetic or a post
hoc description failed outright; seventeen got part right and part wrong, the
last of them §9.20's registration, whose primary was among the parts it got
wrong. One of
those twenty-six
failed by helping: §9.12 registered `nominal` to stay inside ±0.03 and it
improved by 0.040. Another failed on a fact: §9.13's collision bound rested on my
statement that the front end had never collided in any measurement here, and it
had. And two failed for a reason this tally had not yet recorded: §9.16 registered
the co-drift hypothesis through each anchor's *absolute* error, which does not
measure co-drift, so the claims failed and the hypothesis — tested properly in
§9.17, through the drift between the linked pair — held. A registered claim can
be the wrong operationalisation of the idea it was written for, and then a failed
prediction says less than it appears to; the only defence is to state, beside the
claim, why that quantity measures that idea. Confidence
of expression was identical throughout. Three rules came out of them; the
record of each prediction is in [`project_plan.md`](project_plan.md). The
fifteenth also broke this report's own stated practice: its null was registered
as an interval including zero, which intervals of ±0.4 satisfy whatever is
true, so its conclusion rests on a sharper test added afterwards and labelled
as such. The sixteenth was committed to the repository before its data existed,
so its timing is checkable rather than asserted. Two were scored in part on a
Nav2 `noisy_lidar` cell that had no noise in it: §9.4's, which pools that cell,
and §9.5's, which pools it and named it the largest contributor. Re-scored on the
corrected cell (§9.20), §9.4's loses one clause — Nav2's pose error under 0.20 m
on every condition — and §9.5's is unchanged; both stay where they were counted.

**A measurement predicts only where something has been measured.** Carried
into regimes nothing had measured, measurement-derived forecasts failed like
intuition: Nav2 ≥ 0.92 on `dynamic`, extrapolated from static clutter, and
recurrence, forecast from three fixed-window nulls and significantly worse
instead (−0.068, p = 0.017). Crossing a boundary of its own, 4:1 → 1:1, but *with*
the far side already measured, the coverage forecast held on magnitude and
mechanism. A crossing is only extrapolation while the far side is unmeasured —
and careful reasoning does not extend a measurement's reach, it only hides the
overreach. Nor does a measurement support a claim at a resolution it never had:
from 0 of 8 episodes I forecast identical outcomes on all 400, and a few percent
differed. The one prediction that held on every clause was the one whose
quantity had been measured first, on the same worlds, with nothing else in
the loop. By the rule of three, 0 of 8 is compatible with a true rate near 37%.

**A right number is not a right model.** The resolution forecast crossed no
boundary and its number held (+0.032 against |Δ| < 0.03), but its reasoning was
wrong: collisions improved by 0.058 where it said they would not. Only the
discriminator registered beside it showed which. The eighteenth is the same
lesson from the other side: its calculated bound held — the estimate was
0.03 m off at contact, inside the 0.1 m computed — and its conclusion failed,
because the safety margin the bound was compared against was missing from the
plan in force in 15 of 18 collisions. The nineteenth split the ordinary way:
the encoder cost survived re-pricing inside its registered band, and the
failure it was predicted to move into never came.

**A mechanism claim needs a cell where the mechanism should not act.** Churn and
commitment length each fit every number available, and each was refuted by a
control that moved as much as the treatment. Without those cells, +0.070 on
`dynamic_dense` and a tidy horizon optimum would have read as confirmations.
Each cost one extra condition.

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
- **Nav2's `noisy_lidar` cells were first measured without the noise.** The
  bridge configured the condition's range noise and never applied it, so four
  runs scored Nav2 on a clean sensor while the hand-written stack read a noisy
  one. All four were repeated with the noise delivered (§9.20), and the bridge
  now measures what it delivers. The corrected full-privilege passes ran at 5×
  real time while §4.1's other ten ran unthrottled, because the unthrottled
  harness now abandons episodes on a clean sensor; the published passes show no
  sign of that fault, but the protocol behind them is fragile and any re-run
  should use the cap.
- **§9.4's SLAM arms ran one pass each**, held to 5× real time, against the
  full-privilege arm's two unthrottled passes. And §9.5 ran this stack at 360
  beams on parameters tuned for 32, deliberately: its mapper's log-odds
  weights were set for a sparse scanner, and what that costs under noise is
  part of what it measured.
- **The localisation of §9.3 is a front end with no back end**: no pose
  graph, no loop closure, nothing that revisits a past estimate. Its worst
  failure — matching doing more harm than dead reckoning on `sparse` --
  is the failure that a back end exists to prevent, so that number bounds
  this design rather than scan matching.
- **The mapping stack of §9.2 is basic**: an occupancy grid, A\*, pure
  pursuit and the standard recoveries, built for this comparison and fixed
  after its first run measured five faults in it rather than the cost of
  mapping. Its numbers bound what *this* stack pays, not what mapping costs.
- **Single-seed findings are flagged as unreplicated** throughout, including
  the one significant reward-ablation result (`step_penalty` improving nominal
  SPL by +0.066).

## 12. Future work

In order of expected information per GPU-hour:

1. **Make the robot able to drive a tight gap, not merely willing to plan
   one.** Four repairs have been rejected on val — commitment in distance, in
   time, refusing to plan through the unknown, and relaxing the clearance —
   and the last is the one that says where to go next. It moved the robot from
   6.29 m short to 4.80 m short on the eleven worlds §9.9 identified, closing
   more than half a metre on 8 of 11, and converted none of them into
   arrivals while collapsing the stack's zero-collision record to 12 crashes
   (§9.10). Clearance was genuinely blocking those worlds and is not
   sufficient to pass them: a 0.22 m robot in a gap under 0.40 m has no room
   for the 0.07–0.22 m of pose error §9.5 measured. So the work is on the
   estimator and the controller — a back end for the pose (item 3), and
   measuring cross-track error against gap width directly, which is a
   diagnostic this project has never run and which would say whether the
   robot misses the gap because it does not know where it is or because it
   cannot hold a line. Two cheaper things remain open: costing unknown cells
   *above* free ones rather than refusing them outright, the graded middle
   between the arms §9.8 measured, and varying the step budget, since all 39
   failures there spend the full 500 steps with no collision at all, some of
   them 0.96 of the way to the goal — a limit this comparison has never
   varied. None of this touches the seven clutter failures on worlds that had
   a margin-safe route all along, which remain unexplained.
2. **Under noise, what a stack does with the pose error it cannot remove.**
   Localisation under noise is no longer a gap to close: with the noise
   delivered, slam_toolbox ends episodes 0.220 m from the truth and this stack
   0.228 m (§9.20), and the six levers spent on this stack's drift — clearing
   scaled by beam count (§9.14), a cleaner map for closures that never read it,
   a noise-aware closure match (§9.15), closures reweighted or filtered in the
   solve (§9.17), the motion prior (§9.18), and slam_toolbox's own front end
   (§9.19) — were aimed at a gap a production stack does not close either.
   What remains is arrival: 0.950 for Nav2 against 0.890, eight worlds to two,
   not resolved at a hundred worlds (p = 0.11). More worlds would say whether
   it exists; if it does, it lies in the planner and
   the controller tolerating a fifth of a metre of error, which is item 1's
   question asked in open space, and the same cross-track diagnostic answers
   both.
3. **A better front end for the sparse sensor.** The back end is done and
   measured on the held-out worlds (§9.13): pooled +0.010 over 600 paired
   episodes, 14 won and 8 lost, clutter pose error down 20%, not enough to
   replace the published front end. What it settled is where the 32-beam
   problem lives. On val the front end there is already 0.472 m lost, and a
   graph can only redistribute error, not invent information the scans never
   carried; on held-out the back end's pose error at 32 beams got slightly
   *worse*. A sparse sensor therefore needs more geometry per scan — matching
   against features or line segments rather than a likelihood field of
   points — rather than more inference over poses.
4. **An encoder that can measure a gap.** The RGB features carry the
   clearance ahead as well as a depth vector does and the *width* of the gap
   past it not at all (§8.5). The first convolution strides 4 across a
   64-pixel-wide image, so a gap two columns wide survives as at most half a
   feature; a narrower stride, or a wider render at the same field of view, is
   one training run per arm, and the probe says in advance what to measure
   rather than waiting for success rates to move.
5. **A recurrent policy that was actually tuned.** §9.1 tested one and it was
   worse everywhere, but it ran on hyperparameters chosen for an MLP so the
   comparison would be algorithm-only. That makes the result a statement about
   dropping recurrence into this setup rather than about recurrence, and it
   cost 19× the wall clock for the same sample budget. A sequence length, an
   LSTM width and a learning rate chosen *for* the recurrent arm would say
   whether the idea or the transplant failed. Lowest expected value of the
   five: the control cell says the deficit is not about motion at all.
6. **Harder perception** — texture, lighting variation, sensor artefacts — to
   turn the encoder-cost lower bound into an estimate.
7. **Sim-to-real** on a TurtleBot-class base. The action space is already
   `Twist`, so the policy transfers without modification.

## 13. Reproducing

```bash
pip install -e ".[dev,viz]"
pytest                                        # 541 tests
python scripts/check_docs.py                  # every doc link resolves
python -m vision_nav.training.train           # privileged RL
python scripts/run_benchmark.py --rl <model>  # comparison matrix
python scripts/perception_audit.py            # §7, no training required
python scripts/seed_analysis.py --arm ...     # §6.2, §7.1, §8
```

The recurrent arm of §9.1 needs one optional dependency, and its two result
files come from two scripts:

```bash
pip install -e ".[recurrent]"                 # sb3-contrib, for RecurrentPPO
python -m vision_nav.training.train env=nav_dyn_fast algo=ppo_lstm \
    train.run_name=dynfastrec_s0 train.seed=0
python scripts/fast_movers_experiment.py --arm memoryless=... --arm recurrent=...
python scripts/recurrence_ablation.py         # is the memory actually used?
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
