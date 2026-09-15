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

**All dynamic numbers here use block-triggered replanning**, which wins on all
four dynamic cells and is identical to the previous best on the six static
ones, where a correct map means the path is never blocked and it never fires.
That moves the published comparison against the learned policy: `dynamic_dense`
was −0.040 and not significant against the timed baseline, and is −0.110 at
p = 0.031 against this one.

Three narrowings, each from improving the baseline rather than from new
evidence about the policy. The full chronology is in
[`project_plan.md`](project_plan.md); §10 draws the lesson.
