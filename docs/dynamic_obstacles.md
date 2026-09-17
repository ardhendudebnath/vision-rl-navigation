# Where the map is wrong: the moving-obstacle study

A companion to [`report.md`](report.md), which carries the conclusions. This
carries the argument.

Adding obstacles that move and are absent from the map is the first place a
reactive policy has a structural reason to beat a map-based planner. Chasing
that took eight phases, produced three successive corrections to the same
claim, and ended somewhere other than where it started — so it is written out
in full here rather than compressed into the main report, where it would have
crowded out six phases of perception work.

The short version, for a reader who wants only that:

- The learned policy reaches parity with the planner on sparse movers and
  nowhere else.
- That parity is the **planner degrading under motion**, not the policy
  coping — freezing the movers restores the classical advantage in full.
- Real Nav2 beats the hand-written baseline under clutter, and the hand-written
  baseline beats the learned policy once given its best replanning policy.
- Four candidate mechanisms for the motion cost were tested and eliminated;
  it remains unexplained.
- Frame stacking looked inert across three encodings and a 3× speed range.
  It is not: **the reward was setting what the information was worth.**

Everything below is the evidence for those five lines. The chronological
record, including what was believed when, is in
[`project_plan.md`](project_plan.md) phases 3g, 3h and 5b–5h.

## The finding, as first measured

Every condition before this hands the classical planner a **perfect, current,
static map** — its largest privilege and the one real deployments lack.

| Condition | Classical | Learned (best) | Δ | Significant? |
|---|---|---|---|---|
| narrow (static clutter) | **0.850** | 0.682 ± 0.060 | −0.168 | **yes**, p = 0.031 |
| dynamic | **0.880** | 0.860 ± 0.033 | −0.020 | no, p = 0.219 |
| dynamic_dense | **0.820** | 0.710 ± 0.042 | −0.110 | **yes**, p = 0.031 |

Two hypotheses were pre-registered and both failed: that zero-shot policies
would beat the planner, and that frame stacking — which gives the policy
velocity information the map-based stack structurally cannot have — would tip
it. The classical rows use block-triggered replanning, its best configuration
on every dynamic condition; against the timed baseline published earlier they
read 0.870 and 0.750, and `dynamic_dense` looked like parity (−0.040,
p = 0.125). It is not.

## A near-miss in the baseline

The planner replans against a costmap containing the movers, because a
baseline that drove blind into them would prove nothing. Sweeping that
interval showed replanning helps where movers exist (+0.06 to +0.07) and
*hurts* where they do not, dropping `narrow` from 0.850 to 0.690 through path
churn in tight corridors.

With one global setting, `narrow` would have read classical 0.690 against the
policy's 0.682 — 4 of 6 seeds above the baseline, a clean "parity in tight
corridors" claim that was **purely an artefact of a handicap introduced in the
name of fairness**. It survived only because the interval was swept rather
than assumed. The comparison gives the planner its best configuration on every
condition.

The next section later found a better setting still — replanning when the path is
actually blocked rather than on a timer — which dominates both on every
dynamic condition and reduces to `replan_every=0` on the static ones. The
dynamic rows in this report use it.

## What the parity actually is

The parity above is the most RL-favourable result in this report, so it got
the most adversarial follow-up: a production baseline, a controlled
subtraction, and a causal test of the explanation. It survives on `dynamic`
and nowhere else, and it does not mean what it first appeared to.

**Against a production stack it holds on sparse movers only.** Nav2 over two
passes reaches 0.840–0.870 on `dynamic` — indistinguishable from the learned
policy, which is a stronger statement than the original, since the policy now
matches *both* classical stacks. On `dynamic_dense` Nav2 reaches 0.860–0.880
against the policy's 0.710, with all six training seeds below both passes.

**Freezing the movers says why.** Each dynamic condition was re-run with the
movers parked at the positions they already occupy at t = 0 — identical worlds
and seeds, movers still absent from the map, so the map is exactly as wrong as
before and only the motion is gone.

| Clutter | Actor | Frozen | Moving | Cost of motion |
|---|---|---|---|---|
| sparse | classical | 1.000 | 0.880 | −0.120 |
| | Nav2 | 0.985 | 0.855 | −0.130 |
| | learned (best) | 0.900 | 0.852 | **−0.048** |
| dense | classical | 0.980 | 0.820 | −0.160 |
| | Nav2 | 0.945 | 0.870 | −0.075 |
| | learned (best) | 0.778 | 0.710 | **−0.068** |

With the movers frozen the classical advantage returns and widens — −0.100 on
sparse and −0.202 on dense, 0 of 6 seeds above the baseline in both
(p = 0.031), against −0.020 and −0.110 with them running. **The parity is the
planner degrading, not the policy coping.** The learned policy is behind in
every regime; it simply degrades less, costing 0.048–0.068 to motion where the
planner costs 0.120–0.160. A policy that never commits to a path has no plan
to invalidate. That is robustness by *absence of commitment*, not competence
at anticipation — which is exactly what predicts frame stacking buying
nothing.

Freezing had to be done after world generation rather than by zeroing the
amplitude in the config: mover placement validates the swept path, so a zero
sweep accepts positions the moving config rejects and the worlds end up with
different obstacles. A test asserts frozen and moving worlds are identical per
seed, `l*` included, so SPL stays comparable.

**The obvious explanation is wrong, and the control cell is how we know.**
Path churn — the baseline re-committing to a fresh plan every second, the
pathology the near-miss above measured costing `narrow` 0.160 — fits everything: churn is
0.058 m in dense+moving against 0.026–0.038 elsewhere, it is not merely a
function of replan count, and within every cell the episodes that collided
churned more. Removing it tests it. Block-triggered replanning, which rebuilds
only when a mover actually obstructs the path, cuts replans from ~17 per
episode to ~1. Pre-registered: `dynamic_dense` recovers by ≥ +0.05, the frozen
cells stay within ±0.03.

| Cell | Timed | Block-triggered | Δ |
|---|---|---|---|
| sparse moving | 0.870 | 0.880 | +0.010 |
| sparse frozen | 0.980 | 1.000 | +0.020 |
| dense moving | 0.750 | **0.820** | **+0.070** |
| dense frozen | 0.910 | **0.980** | **+0.070** |

The treated cell moved exactly as predicted and **the control cell moved by
the same amount**, leaving the cost of motion unchanged. Churn is real and is
caused by timed replanning, but it is a *clutter* pathology with nothing to do
with whether obstacles move. Without the control, +0.070 on `dynamic_dense`
would have read as clean confirmation of a mechanism the data refutes.

The obvious remaining candidate was that **committing to any plan is itself
the cost** — a trajectory is computed against a snapshot and goes stale the
moment the world moves, so a longer commitment should be worse. DWB's rollout
horizon is exactly that commitment length, and sweeping it over a 6× range
tests it directly, again with the frozen arm as control:

| DWB horizon | Moving | Frozen (control) | Cost of motion |
|---|---|---|---|
| 0.5 s | 0.780 | 0.830 | −0.050 |
| 1.0 s | 0.910 | 0.960 | −0.050 |
| 1.5 s | 0.870 | 0.960 | −0.090 |
| 3.0 s | 0.690 | 0.740 | −0.050 |

**The cost of motion does not move**: −0.050 at every horizon but one, a range
of 0.040 that sits inside the noise band. Meanwhile the control swings by
0.220 — an inverted U with its optimum near 1.0–1.5 s, driven by collisions
that rise to 0.280 (moving) and 0.220 (frozen) at 3.0 s. Horizon length is a
strong determinant of navigation competence and **no determinant at all of
robustness to motion**. Read alone, the moving row would have supported an
optimum-horizon story; the control shows that is ordinary controller tuning.

So the motion cost stands unexplained, with a third mechanism eliminated.
Ruled out: churn, planning failure (A\* never fails to find a route), sensing
(the movers are fully visible to the baseline's costmap), and commitment
length. What survives is only the observation that Nav2's local layer halves
the cost under clutter (−0.075 against −0.160) for a reason none of the four
candidate mechanisms explains.

## Faster movers, explicit velocity, and the reward

Three explanations were offered for frame stacking doing nothing
([Phase 3h](project_plan.md)), and the leading one was that the movers are too
slow to be worth anticipating:
0.15-0.45 m/s against a 0.6 m/s robot. dynamic_fast raises them to
0.8-1.5 m/s, which **outruns the robot**, on worlds that are geometrically
identical seed for seed -- speed is drawn after the placement test and only
feeds the angular rate, so nothing else changes. Two arms trained from scratch
on fast movers, rame_stack 1 and 4, six seeds each, with the slow condition
kept as a control.

| Condition | stack1 | stack4 | Delta | p (exact) |
|---|---|---|---|---|
| **fast** (primary) | 0.652 +/- 0.052 | 0.625 +/- 0.058 | **-0.027** | 0.442 |
| slow (control) | 0.802 +/- 0.027 | 0.778 +/- 0.059 | -0.023 | 0.502 |

**Inert at three times the speed**, and inert by almost exactly the same
amount as on the slow condition. The speed explanation is dead.

This is not a ceiling or floor artefact: tripling mover speed costs every
actor real performance -- 0.880 to 0.750 for the classical planner (collisions
0.110 to 0.250) and about 0.150 for both learned arms. The condition bites
hard; velocity information simply does not help against it.

**Nor is it an extraction problem.** Stacked scans contain velocity only
implicitly, so the remaining reading was that an MLP cannot recover it from
raw ranges. obs_velocity hands the policy the per-beam range delta directly,
against rame_stack=2 carrying the same information at nearly the same width
(133 against 138 dimensions):

| Arm | fast | slow (control) |
|---|---|---|
| no velocity (rame_stack=1) | 0.652 ± 0.052 | 0.802 ± 0.027 |
| implicit (rame_stack=2) | 0.637 ± 0.043 | 0.785 ± 0.060 |
| implicit (rame_stack=4) | 0.625 ± 0.058 | 0.778 ± 0.059 |
| **explicit delta channel** | **0.678 ± 0.034** | **0.820 ± 0.028** |

The explicit channel beats frame stacking by +0.042 on fast (p = 0.108) — and
by +0.035 on the slow control (p = 0.249). Neither is significant, and more to
the point the two are the same size, so this is a mild preference for the
encoding rather than anything being *used* for anticipation. Against having no
velocity information at all it is worth +0.026. Every arm sits within 0.053 of
every other, against per-arm seed spreads of 0.03–0.06.

So velocity information does not help this policy however it is supplied:
implicit at two frames, implicit at four, explicit as a difference channel,
at either speed. Both readings of the Phase 3h null have now been tested and
both are dead.

**But the reward was hiding it.** Phase 3h's third explanation was that the
reward suppresses commitment: a collision costs 20 and a full 500-step timeout
costs 0.01 x 500 = 5, so crashing is four times worse than stalling, and
Section 5.3 already showed the policy optimising that by learning to stop. A
policy that will not act on a prediction has no use for one. Setting the
collision penalty to 5 makes the two costs *exactly equal* — indifference, not
a thumb on the scale — and changes nothing else.

| Reward | stack1 | stack4 | Δ | p (exact) |
|---|---|---|---|---|
| 4:1 (collision 20) | 0.652 ± 0.052 | 0.625 ± 0.058 | −0.027 | 0.442 |
| **1:1 (collision 5)** | 0.660 ± 0.019 | **0.693 ± 0.020** | **+0.033** | **0.019** |
| 1:1, slow control | 0.808 ± 0.021 | 0.807 ± 0.048 | −0.002 | 1.000 |

**Frame stacking works once the reward stops punishing commitment** — 6 of 6
seeds, significant, and absent on the slow control where there is less to
anticipate. The swing between rewards is +0.060.

The outcome breakdown says what actually changed, and it is not that the
policy suddenly learned to anticipate:

| Arm | Success | Collisions | Timeouts |
|---|---|---|---|
| 4:1 stack1 | 0.652 | 0.300 | 0.048 |
| 4:1 stack4 | 0.625 | **0.282** | **0.093** |
| 1:1 stack1 | 0.660 | 0.333 | 0.007 |
| 1:1 stack4 | **0.693** | **0.302** | 0.005 |

Stacking cuts collisions under **both** rewards — by 0.018 at 4:1 and 0.031 at
1:1. The information was being used all along. What differs is what it is
spent on: at 4:1 the collision saving is more than swallowed by timeouts
nearly doubling and net success *falls*; under indifference the same saving
flows straight into successes. **The reward does not decide whether the policy
can anticipate. It decides what anticipation is for.**

That is Result 2 one level up. Section 5.3 found the reward makes the policy
stall rather than get through; this finds it also converts *additional
information* into additional stalling. An observation channel is worth only
what the objective lets the policy do with it.

So The account above needs correcting rather than confirming. Absence of
commitment is real, but it is **caused by the reward rather than intrinsic to
the policy** — and across two rewards, three encodings and a 3× speed ratio,
that is now supported rather than merely last standing.

The pre-registered magnitude was +0.05 and the effect is +0.033, so the
prediction failed on size while getting the mechanism and its specificity
right. That is a different kind of miss from the previous six.

## Recurrence: worse everywhere, and not about motion

Three ways of handing the policy motion information were inert: frame stacking
at two and four frames, and an explicit per-beam velocity channel. Recurrence
was the fourth and last — an LSTM integrates over the whole episode rather than
a fixed window, and *learns* what to keep instead of being handed a
hand-designed summary. Same worlds, same seeds, same 1.5M-step budget; PPO
becomes RecurrentPPO with a 256-unit LSTM for actor and critic, and every
other hyperparameter is copied from the baseline unchanged.

Pre-registered: no effect, |Δ| < 0.03 and p > 0.05, with slow movers as the
control cell.

| Condition | memoryless | recurrent | Δ | p (exact) | seeds higher |
|---|---|---|---|---|---|
| fast movers (primary) | 0.652 ± 0.052 | 0.583 ± 0.022 | **-0.068** | **0.017** | 1/6 |
| slow movers (control) | 0.802 ± 0.027 | 0.723 ± 0.045 | **-0.078** | **0.004** | 0/6 |

**Recurrence is significantly worse, and the control is worse by more.** The
prediction failed on direction, not just on magnitude.

**And that control refuses the motion reading.** If recurrence were failing to
help *with motion*, the slow condition — where there is less to anticipate —
should have been the one it left alone. Instead the harm is slightly larger
there. Whatever the LSTM costs, it is not paid on anticipation: it is a general
property of the arm. This is the third time in this study that a control cell
has stopped a mechanism claim, after churn and commitment length, and the
lesson is the same each time — the experiment answers a narrower question than
the one it was built to ask.

The outcome breakdown rules out the one alternative that would have made this
a story about the reward:

| Arm | Success | Collisions | Timeouts |
|---|---|---|---|
| memoryless, fast | 0.652 | 0.300 | 0.048 |
| recurrent, fast | 0.583 | **0.323** | **0.093** |

Frame stacking at 4:1 cut collisions and paid for it in timeouts — information
used, and spent on stalling. Here **both rise**. The recurrent policy crashes
more *and* stalls more, so it is not trading one failure for another; it is
worse at both, and the reward is not hiding anything.

### Two ways this could have been an artefact, both closed

A negative result is only worth as much as the things ruled out before it was
read, and two failure modes here produce exactly this number.

**Under-training.** RecurrentPPO has more parameters and a harder optimisation
than PPO, so "memory does not help" and "the LSTM had not finished" predict the
same success rate. Comparing each seed's final third of validation evaluations
against the third before it: the baseline gains +0.026 and the
recurrent arm -0.008. Both are inside the ±0.03 band measured across
the four arms that have run this budget to completion. The recurrent arm had
stopped improving — it is converged, not truncated.

**An LSTM that ignored its own memory.** If the policy had learned to route
around its recurrent state, the entire cost would be the architecture and the
recurrence itself would be irrelevant. Evaluating each policy twice on
identical worlds — once carrying state across the episode, once clearing it at
every step — gives 0.583 with memory against 0.268 without
it, a gap of **+0.315**. Memory is emphatically in the loop. The
policy depends on its history; the history simply does not buy performance
that the memoryless arm did not already have.

That second check also prices a near-miss in the harness. Carrying recurrent
state through evaluation had to be *added* for this experiment, and had it been
left out, the recurrent arm would have scored about 0.268 against
0.652 — a Δ near -0.38 that reads as a catastrophic
failure of recurrence and is entirely an evaluation bug. The measured answer
and the artefact point the same direction, which is exactly why the artefact
would have been believed.

### What this does and does not establish

It establishes that **at a matched sample budget, on this task, a recurrent
policy is worse everywhere than a memoryless one**, and that the gap has
nothing to do with moving obstacles.

It does not establish that memory cannot help here. The LSTM ran on
hyperparameters chosen for an MLP — deliberately, so the comparison would be
algorithm-only, but that makes this a result about *dropping recurrence into
this setup*, not about recurrence. The honest summary is that the cheap version
of the idea does not work, and costs **19× the wall clock**
to find out: 78 steps/second against 1480 for the baseline,
for the same 1.5M samples. Sample-matching is the generous choice here, not the
strict one — at equal compute the baseline would have seen far more data, and
the gap would be wider.

The absence-of-commitment account is neither supported nor damaged by this.
Phase 5h remains the load-bearing result: the reward decides what extra
information is worth, and nothing here touches that.

## Knowing where the movers are going: half the motion cost

Four mechanisms for the classical stack's motion cost were eliminated --
replanning churn, planning failure, sensing, commitment length -- and every
actor still treated a mover as a static snapshot at its current position. A
planner that cannot tell a mover approaching its path from one leaving it
replans too late for the first and needlessly for the second.

The adopted baseline was given **oracle** knowledge of motion: the trajectories
are analytic, so it plans around, and triggers replans on, each mover's exact
swept region over the next H seconds. That is an upper bound on what any real
velocity layer could supply. The initial plan stays static-only, so the arms
differ in how they replan and nothing else.

Two guards came first, because the change touched the world's geometry. At
H = 0 the committed churn experiment reproduces **episode-for-episode across
800 episodes**, churn included to the last bit, so every earlier result is
untouched. And the frozen cells are the control stated as an **identity**: a
frozen mover's swept region at any horizon is its current disc, so frozen
episodes must be bit-identical across horizons. They are, at every horizon.

At H = 0 the motion cost is +0.160 on `dynamic_dense` and
+0.120 on `dynamic` -- exactly the figures published
before, which is itself a check.

| `dynamic_dense` | success | of the cost | collision | timeout | p (McNemar) |
|---|---|---|---|---|---|
| 1 s | +0.020 | 13% | -0.030 | +0.010 | 0.5000 (2 won, 0 lost) |
| 2 s | +0.070 | 44% | -0.100 | +0.030 | 0.0156 (7 won, 0 lost) |
| 4 s | +0.040 | 25% | -0.080 | +0.040 | 0.3438 (7 won, 3 lost) |

**At two seconds, prediction recovers 44%
of the dense motion cost** -- +0.070, significant
at the Bonferroni-corrected α = 0.05/3, with every one of the
7 episodes that changed outcome changing for the
better. It works the way the hypothesis says it should: collisions fall
-0.100. The planner was driving into movers
it failed to anticipate.

### What bounds the claim

**It is significant by a hair, and it is the best of three.** p =
0.0156 against a corrected threshold of 0.0167, and neither
neighbouring horizon reaches significance. Phase 5e read a tidy optimum across
a sweep as confirmation and its control refuted it; the correction was
registered in advance for exactly that reason, and the result survives it, but
not by much.

**`dynamic` agrees in direction and not in significance.** Its best is
+0.060 at 4 s (p = 0.070,
7 won, 1 lost),
recovering 50%. With twelve
episodes of cost to recover from rather than sixteen, that is as consistent
with low power as with no effect, and it is reported as unconfirmed rather than
as a null.

**The ceiling needs stating carefully, and the first writeup did not.** An
oracle bounds the *quality* of the information, not how it is used. So this
bounds how much better velocity estimates could help *this* planner -- which
consumes them as swept regions and plans in space -- at about half. It does not
bound what a planner reasoning in space-time could do with the same input. The
first version of this paragraph said velocity explains "at most about half" the
motion cost outright, which reads a limit of the architecture as a limit of the
information.

Replanning rises about 4-fold
(0.9 replans an episode at H = 0
against 3.8 at 2 s) while success
rises. Phase 5d found churn does not cause the motion cost, and this agrees
from the other side: more replanning is harmless when it replans against the
right thing.

### The other half: prediction everywhere else the stack reads movers

The oracle above fed replanning and the replan trigger. Two other places still
read movers as snapshots: the **initial plan**, left static-only so the arms
differed in replanning alone, and the controller's **reactive slow-down**,
which reads clearance to where a mover currently is. Each was given the same
2 s oracle, measured against the replanning-only arm, whose motion cost
remaining is +0.090 on both conditions.

"The initial plan sees movers" is two interventions and was tested as two.
Seeing movers *at all* changes frozen worlds too, since a frozen mover is an
obstacle the map lacks; seeing *where they are going* changes only moving ones,
and is bit-identical on frozen cells, as is prediction in the slow-down. Both
identities hold, and the replanning-only arm reproduces the table above on
every rate.

| added to replanning prediction (`dynamic_dense` unless noted) | success | collision | timeout | episodes |
|---|---|---|---|---|
| initial plan sees movers where they are | +0.010 | +0.000 | -0.010 | 2 won, 1 lost |
| ... and where they are going | +0.000 | +0.020 | -0.020 | 0 won, 0 lost |
| slow-down reads where they are going | +0.000 | +0.010 | -0.010 | 1 won, 1 lost |
| slow-down reads where they are going, `dynamic` | -0.040 | +0.040 | +0.000 | 1 won, 5 lost |

**None of it recovers anything.** Oracle knowledge of every mover's future,
supplied at every point this planner and controller read movers, recovers half
the motion cost and not a percentage point more. Predictive slow-down is, if
anything, counter-productive on `dynamic` -- more collisions, the one
direction a caution rule should not move -- though at
p = 0.22 that is not a finding. Hesitating for a mover
about to arrive is plausibly exactly wrong when the better move is to pass
before it does.

So the remainder is not fixed by more information *within this architecture*.
Two explanations are left, and they are separable: a planner that uses the same
oracle in space-time rather than as swept regions, or physical limits on
evading a mover the robot has correctly anticipated. The second is testable by
raising the robot's speed and acceleration limits and nothing else.

### Is the rest physical? Doubling the robot's agility

Two explanations survived for the remaining half: this planner uses a perfect
input crudely, or the robot cannot physically evade a mover it has correctly
anticipated. The second was tested by scaling the robot's velocity *and*
acceleration limits together -- 0.75× to 2× -- so time-to-top-speed is unchanged
and only agility differs, over all 200 test episodes.

One fact shaped the expectation before anything ran. The `dynamic_dense` movers
peak at 0.15-0.45 m/s, and only momentarily, while the robot does 0.6 m/s at
1×: it already outruns every mover.

Three confounds were handled rather than assumed away. A faster robot meets
fewer movers, so episode length was logged. Controller distances are tuned at
0.6 m/s, so a speed whose frozen-mover success falls more than 0.03 below 1× was
declared uninterpretable in advance. And the replan trigger's look-ahead was
scaled with speed so anticipation *time* stayed constant. At 1× the run
reproduces Phase 5m on its first 100 episodes, and the frozen identity holds at
every speed.

| `dynamic_dense` | steps | cost | remaining with oracle | remaining, vs 1× (95% CI) | controller valid |
|---|---|---|---|---|---|
| 0.75× | 238 | +0.220 | +0.170 | [+0.030, +0.145] | **no** |
| 1.0× | 188 | +0.145 | +0.085 | — | yes |
| 1.5× | 132 | +0.150 | +0.105 | [-0.030, +0.070] | yes |
| 2.0× | 98 | +0.135 | +0.090 | [-0.045, +0.055] | yes |

**Doubling agility leaves the remaining cost where it was** --
+0.005, with an interval of [-0.045, +0.055].
Removing it entirely would take −0.085, and
the interval excludes removing more than about half. Time spent among movers
halves, from 188 to
98 steps, and the robot can evade twice as
fast; neither touches it. A physical limit on evasion is not the main
explanation for the dense remainder.

Three things this does **not** show. `dynamic` is inconclusive: the remainder
falls -0.030, not significantly, with an interval
[-0.085, +0.025] that cannot exclude agility removing nearly
all of it. The motion cost does not detectably shrink with halved exposure
either (-0.010 on dense), but that interval,
[-0.070, +0.050], is too wide to call the cost insensitive to
exposure. And 0.75× on dense fails its validity check -- a slow robot times out
in clutter -- so its rise is not a result. What survives is the planning
explanation, which is a candidate left standing, not a confirmed one.

**The oracle's benefit replicates at every speed**: +0.045
to +0.060 on dense (p 0.004 to
0.078) and +0.045
to +0.060 on sparse, all significant but dense
at 1.5×. These are paired comparisons within a speed, so the validity rule, which
concerns the motion cost across speeds, does not bear on them. On all 200 episodes the Phase 5m effect is p = 0.004,
where on its 100 it was a marginal 0.016.

**The pre-registered criterion was the weak one, and the decisive test is post
hoc.** It was registered as "the 2× − 1× recovery interval includes zero". Those
intervals came out near ±0.4 -- wide enough to include zero *and* the +0.25 a
physical limit was predicted to show -- so the criterion would have passed
whatever was true. That is an unbounded null, the form Phase 3d warned against
and this project lists as avoided practice. The interval on remaining cost above
is sharper and is what the conclusion rests on, but it was added after the first
run and is labelled so throughout; the registered prediction is unchanged.

### A planner that reasons about time

What survived every test so far was that a planner reasoning only in space
uses a perfect input crudely: it can route around where a mover goes, but not
wait for it or pass ahead of it. A space-time A\* over (row, col, time), held
to optimality against brute force and with each safety rule mutation-tested,
was paired with an agent that tracks a *schedule* rather than a path -- pure
pursuit would drive straight through a planned wait -- and shares every
controller parameter with the spatial baseline.

That agent differs from the baseline in five ways, not one, so beating it could
not be credited to timing. The test is an ablation: the same agent shown every
mover's union over the planning window at every step, so it knows *where*
movers go but not *when*. On frozen worlds the two are bit-identical, as they
must be. The prediction, its decision rule, and a null bounded at ±0.03 were
pushed to the repository before the run finished.

| success, 200 episodes | spatial | swept (no timing) | full (timing) |
|---|---|---|---|
| `dynamic`, moving | 0.905 | 0.995 | 0.945 |
| `dynamic`, frozen | 1.000 | 0.995 | 0.995 |
| `dynamic_dense`, moving | 0.890 | 0.910 | 0.960 |
| `dynamic_dense`, frozen | 0.975 | 0.985 | 0.985 |

**On dense clutter, knowing *when* matters** -- full against swept
+0.050, p = 0.031, 14 episodes won
and 4 lost, interval [+0.010, +0.090]. It meets the
registered rule. **But not by the registered mechanism.** The prediction said
fewer collisions; the gain is in timeouts, −0.060, while
collisions move +0.010. Blocking every cell a mover will
touch for the whole window walls off corridors, and the swept planner gets
stuck; knowing when those cells clear un-sticks it.

**On sparse worlds, knowing *when* is significantly worse** --
−0.050, p = 0.002, 0 won and
10 lost, all of it collisions (+0.050).
The likely cause, not yet tested: the full planner threads gaps a single
0.24 s plan step ahead of a mover with no temporal safety margin, so any lag in
tracking puts the robot where the mover arrives. The swept planner never threads
a timing gap, and on a sparse world it has room to go round.

The registered decision rule labelled that result "inconclusive", because it
named only MATTERS and a bounded INERT. So did the code that applied it -- the
same defect fixed in `fast_movers_experiment.py` earlier in this project,
written again. The rule now has a HARMS outcome, added after the run and
labelled as such, with a test that a significant harm is never called
inconclusive.

**And the ablation stopped a misattribution.** Against the spatial baseline the
swept agent -- no timing at all -- closes the sparse motion cost completely
(+0.095 to +0.000,
18 won and 0 lost). The window, replan
timer and schedule tracking do that, and a comparison of the full agent against
the spatial baseline alone would have credited it to reasoning about time. On
dense clutter those same differences trade collisions for timeouts and net
nothing (+0.020, p = 0.58); timing turns the
timeouts into successes. Together they cut the dense motion cost remaining after
oracle prediction from +0.085 to
+0.025.

So timing buys access at the price of robustness. It pays where a cautious
planner would be walled in and costs where it would not. A planner that keeps a
temporal margin -- or chooses between the two by how blocked a corridor is -- is
the obvious next test, and the sparse harm is the prediction it has to beat.

### A temporal safety margin

The cause offered for timing's harm on sparse worlds was that the planner
threads gaps one 0.24 s plan step ahead of a mover, so any lag tracking the
schedule puts the robot where the mover arrives. The test widens each mover's
occupancy by a margin of plan steps either side. A frozen mover is the same at
every step, so widening it changes nothing: margins 0 and 4 are bit-identical on
frozen worlds, and the margin-0 and swept arms reproduce the previous
experiment episode for episode. The prediction was committed before the run,
with the primary margin fixed at two steps to match the tracker's 0.5 s lead.

| margin | sparse success | sparse collisions | dense success |
|---|---|---|---|
| 0 (Phase 5p) | 0.945 | 0.050 | 0.960 |
| 1 step | 0.980 | 0.015 | 0.970 |
| 2 steps | 0.995 | 0.000 | 0.975 |
| 4 steps | 0.995 | 0.000 | 0.970 |
| swept, no timing | 0.995 | 0.000 | 0.910 |

**Two steps removes the harm completely.** Against no margin it is
+0.050 on sparse worlds, p = 0.002, winning back all
10 episodes and losing none, and the whole of it is collisions
(−0.050). The response rises with the margin and stops at two
steps, which is what the proposed cause predicts and hard to produce otherwise.
The dense gain survives and grows slightly: +0.065 over the
when-blind planner, p = 0.004.

One registered clause failed. Four steps was predicted to make the planner
cautious enough to drift back towards the when-blind one on dense clutter; it
does not (0.970 against 0.910,
still +0.060 ahead).

**So the motion cost is explained.** With oracle trajectories, a planner that
reasons in space-time and keeps a two-step margin loses
0.010 to motion on dense clutter and 0.000 on sparse worlds, where
the spatial baseline lost 0.145 and 0.140. What looked like an
irreducible cost of moving obstacles was a planner that could not reason about
time and, once it could, needed room for its own tracking error.

The condition is not incidental. Every number in this section uses the movers'
exact future positions, which no real robot has, and it says nothing about the
learned policy, which never had them. What it settles is what a better
*estimate* of motion could be worth to this stack: all of the motion cost, if
the estimate is good enough -- which is now the question worth asking.

### A constant-velocity estimate instead of the oracle

Every number in the margin section used the movers' exact futures. This
replaces them with the simplest estimate a real robot could make: each mover
carried forward at the velocity between the last two positions the agent itself
observed. The observations are noise-free, so the error measured is the
model's -- a straight line drawn along a sinusoid -- and not a sensor's. A
frozen mover has zero velocity and so an exact estimate, and the estimating and
oracle agents are bit-identical on frozen worlds; the oracle arm reproduces the
margin experiment on every episode. The prediction and its decision rule were
pushed to the repository before the run, in `d428cc5`.

| 200 episodes | sparse success | sparse collisions | dense success | dense collisions | dense timeouts |
|---|---|---|---|---|---|
| oracle, 2-step margin | 0.995 | 0.000 | 0.975 | 0.010 | 0.015 |
| estimate, 2-step margin | 0.975 | 0.020 | 0.910 | 0.075 | 0.015 |
| estimate, 4-step margin | 0.970 | 0.025 | 0.910 | 0.070 | 0.020 |

**On dense clutter the estimate costs 0.065** against the oracle -- p = 0.001,
14 episodes lost and 1 won, interval [−0.105, −0.030] -- and all of it
is collisions (+0.065), with timeouts unchanged. On sparse worlds it
costs 0.020, all four discordant episodes lost (p = 0.125,
interval [−0.040, −0.005]): not significant, and not bounded-inert either.
The registered rule returns COSTLY. **Doubling the margin recovers none of it**:
four steps against two is +0.000 on dense clutter, interval
[−0.025, +0.030].

**The prediction was that estimation would cost nothing, and it failed.** It
was arithmetic: a straight line drawn along the fastest mover's sinusoid drifts
by at most about 0.1 m over the second executed between replans, inside the
planner's 0.18 m safety margin. The clauses registered for the case it failed
put the loss on dense clutter and said a wider margin would not recover it,
both right, and named timeouts as the channel, which was wrong.

**What the lost episodes share** -- described after the fact, not tested, by
`scripts/estimate_diagnostic.py`. All 18 end in contact with a mover, none with a
wall. The arithmetic was right: at contact, the estimate the plan rested on was
a median 0.030 m from the truth, and at most 0.159 m. What it leaned on
was not there: in 15 of the 18, the plan in force had been made at the bare
robot radius, the last of the planner's three fallbacks, with no spatial margin
at all. A plan with no margin is safe under exact prediction and unsafe under
any error, and a temporal margin cannot supply what a spatial one lacks -- which
fits every number above. But a planner also falls back when a mover is already
close, so the bare-radius plan may be a symptom. The evidence either way is
thin: before each episode's last two seconds, the estimating agent made
reduced-radius plans more often than the oracle on dense clutter (median share
0.116 against 0.000) but not on sparse worlds (0.033
against 0.077), and 6 of the 18 episodes made none at all.
Withholding the bare-radius fallback is the test.

**What survives.** With the estimate, the motion cost is 0.075 on dense clutter
and 0.020 on sparse worlds, against 0.145 and 0.140 for the spatial
baseline without prediction: about half of the oracle's reduction on dense
clutter and most of it on sparse worlds, from noise-free observations. The
explanation of the motion cost stands. What it is worth to a robot without an
oracle is about half as much on dense clutter, before any sensor noise.

### Withholding the zero-margin fallback

The lost episodes left two readings of that fallback: the cause of the contact,
or where a robot already too close ends up. The test withholds the bare radius
from movers -- the static map may still use it for a tight passage -- on the
estimating agent and, as a control, on the oracle, where a zero-margin plan is
safe. Until an episode first reaches that fallback the setting cannot change
anything, and all 745 such episodes across the four cells are bit-identical to
their unfloored twins. The prediction, CAUSE, was pushed before the run in
`df8088e`.

| 200 episodes | reach bare radius, sparse | reach it, dense | dense success | dense collisions |
|---|---|---|---|---|
| oracle | 2 | 6 | 0.975 | 0.010 |
| oracle, fallback withheld | 2 | 6 | 0.970 | 0.015 |
| estimate | 16 | 31 | 0.910 | 0.075 |
| estimate, fallback withheld | 16 | 31 | 0.920 | 0.065 |

**It is a symptom.** The estimate does drive the planner into its last fallback
five times as often as the oracle on dense clutter, 31 episodes against 6. But
withholding it changes 2 dense outcomes, both to wins: +0.010, p = 0.5,
interval [+0.000, +0.025], a bounded null. On sparse worlds, of the 16 episodes
that reach the fallback, not one outcome changes. The oracle control is bounded
null on both conditions, as registered. With the fallback withheld the estimate
still trails the oracle by 0.055 on dense clutter (p = 0.003, 12 episodes
lost and 1 won).

So the post hoc description was right about what the lost episodes shared and
wrong about what it meant. A plan with no margin is what the planner makes once
a mover is already close; the robot gets close earlier, on an estimate acted on
for up to a second between replans. That is the next place to look.

### Replanning when the estimate is contradicted

The fallback was a symptom: by the time the planner reaches it, the robot is
already too close. Between scheduled replans the agent acts for up to a second
on the estimate made at the last plan, so this test replans the moment an
observed mover is more than 0.05 m or 0.02 m from where that estimate put it.
The control is free, because the oracle is never contradicted: the oracle agent
with the trigger is bit-identical to one without on every moving-world episode,
and the trigger never fires. Pre-registered in `b4ccdc2`.

| 200 episodes | sparse success | dense success | dense collisions | replans, dense episode |
|---|---|---|---|---|
| oracle | 0.995 | 0.975 | 0.010 | 24.9 |
| estimate | 0.975 | 0.910 | 0.075 | 24.9 |
| estimate, 0.05 m trigger | 0.960 | 0.925 | 0.055 | 26.5 |
| estimate, 0.02 m trigger | 0.995 | 0.925 | 0.055 | 35.7 |

**The registered rule returns UNRESOLVED.** On dense clutter the 0.02 m trigger
gains +0.015 (p = 0.55, 7 episodes won and 4 lost), interval
[−0.015, +0.050]: neither the gain the prediction needed nor a bounded null.
Two hundred episodes cannot say whether freshness is worth nothing there or most
of the loss. The 0.05 m trigger gains exactly as much with a quarter of the
triggered replans, so the dose-response the prediction asked for is absent too.
What is not in doubt: with the trigger, the estimate still trails the oracle by
0.050 on dense clutter (p = 0.002, 10 episodes lost and none won).

**On sparse worlds fresh estimates close the gap.** The 0.02 m trigger wins back
all four episodes the estimate lost and loses none, and its success matches the
oracle's on every one of the 200 episodes. Four episodes cannot reach
significance -- p = 0.125 is the smallest four discordant pairs allow -- so
this is a direction, not a finding, but it is the direction staleness predicts.

So on dense clutter a wider temporal margin, a withheld fallback and fresh
estimates have each failed to account for the remainder. The explanation named
first, in the prediction for the estimate itself, and never tested, is the far
end of the window: a straight line carried seven seconds out, and held past it
as a permanent obstacle.

### Capping how far the estimate reaches

The one explanation left untested was the far end of the window: each straight
line carried seven seconds out, its last step held past the window as a
permanent obstacle. The test carries the line at most 2 s past the latest
observation, or 1 s, and holds the mover there. A frozen mover's estimate never
moves, and capped and uncapped agents are bit-identical on every frozen episode.
Pre-registered in `231c23b`.

| 200 episodes | sparse success | sparse collisions | dense success | dense collisions | dense episodes reaching bare radius |
|---|---|---|---|---|---|
| estimate, uncapped | 0.975 | 0.020 | 0.910 | 0.075 | 31 |
| capped at 2 s | 0.955 | 0.040 | 0.920 | 0.065 | 21 |
| capped at 1 s | 0.890 | 0.105 | 0.875 | 0.110 | 34 |

**The registered rule returns UNRESOLVED.** On dense clutter the 2 s cap gains
+0.010 (p = 0.73, 5 episodes won and 3 lost), interval [−0.015, +0.040].
It does what the prediction's secondary clause said -- ten fewer dense episodes
fall back to the bare radius -- and it makes no difference to how many succeed,
which is Phase 5t's lesson again.

**The far end is used, not a cost.** Carried only 1 s, the estimate loses
0.085 on sparse worlds (p = 0.0005, 20 episodes lost and 3 won), every
episode of it a collision (+0.085). Why is not measured here -- a planner told
that movers stop a second out can plan through where they will be -- but the
direction is not in doubt: the seven-second line is wrong in detail and still
worth more than no line.

So four explanations of the estimate's dense-clutter cost have been tested from
the planner's side -- a wider temporal margin, the zero-margin fallback, a stale
estimate and the far end of the window -- and none accounts for it. What none of
them changed is the model: a straight line drawn along a sinusoid. That is the
next thing to change.

### Fitting the oscillation instead of drawing a line

Four changes to how the planner uses a straight-line estimate recovered none of
its dense-clutter cost. This changes the line. Every mover here runs
`centre + dir * A sin(w t)`, so each obeys `a = -w^2 (p - c)`, which is linear
in `w^2` and `w^2 c`: three unknowns of least squares over the track the agent
has watched, and the prediction is that equation's exact solution from the
current state. The fit is told nothing about any mover. A frozen mover has no
curvature to read, the fit declines, the agent falls back to the line -- exact
when nothing moves -- and the two agents are bit-identical on every frozen
episode. Pre-registered in `5091da1`, with the estimator's own error measured
first and the commit saying so.

| median error against the truth, `dynamic_dense` | 1 s ahead | 2 s ahead | 7 s ahead |
|---|---|---|---|
| straight line | 0.023 | 0.086 | 1.192 |
| fitted oscillation | 0.0000 | 0.0000 | 0.0000 |

| 200 episodes | sparse success | dense success | dense collisions |
|---|---|---|---|
| oracle | 0.995 | 0.975 | 0.010 |
| straight line | 0.975 | 0.910 | 0.075 |
| fitted oscillation | 0.995 | 0.980 | 0.005 |

**The model was the whole of it.** Against the straight line the fit is worth
+0.070 on dense clutter -- p = 0.0001, 14 episodes won and none lost --
and against the *oracle* it is bounded-inert on both conditions: +0.005
on dense clutter, interval [+0.000, +0.015], and not one episode different
on sparse worlds. The motion cost falls to 0.005 on dense clutter and 0.000 on
sparse, where the oracle left 0.010 and 0.000 and the straight line 0.075 and
0.020.

So the answer to the question Phase 5r opened is that none of the 0.065 belonged
to the planner. A margin, a fallback, a replan trigger and a capped reach each
left it untouched because each was a way of coping with a wrong estimate; the
estimate stopped being wrong and the cost went with it.

**What this is not.** The observations are noise-free and the fitted model is
exactly the one the simulator integrates, which is the most favourable case an
estimator can be handed. The number to take from it is an upper bound: with
perfect sensing and the right model class, a real estimator is worth what the
oracle was worth. What a noisy scan does to a three-second fit is the next
question, and nothing here answers it.

**All dynamic numbers here use block-triggered replanning**, which wins on all
four dynamic cells and is identical to the previous best on the six static
ones, where a correct map means the path is never blocked and it never fires.
That moves the published comparison against the learned policy: `dynamic_dense`
was −0.040 and not significant against the timed baseline, and is −0.110 at
p = 0.031 against this one.

Three narrowings, each from improving the baseline rather than from new
evidence about the policy. The full chronology is in
[`project_plan.md`](project_plan.md); §10 draws the lesson.
