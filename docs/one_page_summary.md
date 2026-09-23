# Learning Vision-Conditioned Navigation Policies — one-page summary

*A comparative study against classical planning.*
Repository: <https://github.com/ardhendudebnath/vision-rl-navigation> ·
Full report: [`docs/report.md`](report.md)

## The question

Learning-based navigation is usually evaluated against weak baselines, on the
distribution it trained on, with a single training seed. Each choice flatters
the learned method. This project asks the unflattering version instead:
**where does a learned navigation policy actually beat a strong classical
planner, where does it lose, and how does each degrade when the world stops
looking like the training set?**

## Setup

A procedurally generated point-goal task with a differential-drive robot,
guaranteed-solvable worlds, and disjoint train / validation / test / OOD seed
bands. Actors are compared on **identical worlds in identical order**, asserted
at evaluation time rather than assumed, and scored with SPL (Anderson et al.,
2018) so the numbers are comparable to the embodied-navigation literature.

Three actors: PPO from sensor observations; a classical A\* + pure-pursuit
stack given the full obstacle map and exact pose; and **real ROS 2 Nav2**
driven over a bridge into the same simulator. The classical stack is
deliberately over-privileged — a baseline that loses through handicap proves
nothing.

## Findings

1. **The planner wins on every condition — while it is handed the map**, by
   margins that grow with clutter (1.000 vs 0.960 success nominally, 0.850 vs
   0.682 in tight corridors). Building that map from the same scan the policy
   reads costs it the clutter margin entirely (0.590 vs 0.600 in tight
   corridors, 0.650 vs 0.640 in dense clutter) and almost nothing in open
   worlds. Taking its *pose* as well — odometry corrected by scan matching
   against that same map — costs nothing further in clutter (0.630 and 0.640)
   and most of what remains in open worlds (0.530 against 0.990 on the largest
   arenas). The two privileges are worth opposite things, and structure is why:
   it is both what a map is needed for and what a pose is recovered from. Nav2
   with SLAM, asked the same question, pays at least as much given the same
   32-beam scanner and a seventh of it given 360 beams; this stack given 360
   beams halves its cost, all of it in localisation. The pose stood in for the
   sensor, the map for the implementation. Four
   standard explanations — insufficient data, wrong training distribution,
   insufficient compute (tested twice, to 2.7× budget), reward
   mis-specification — were each tested and rejected. The cause is
   behavioural: the reward makes a collision cost four times a timeout, so the
   policy correctly learns to stall rather than crash.
2. **A training-free audit of the sensor** quantified a geometric limit, then
   forecast two later training experiments to within 0.021 and 0.001. It
   separates coverage (causal: +0.095, p = 0.024) from angular resolution
   (inert in success: ±0.003 against a pre-registered ±0.01 bound). Resolution
   is not behaviourally inert, though: whether it raises or lowers collisions
   is decided by the reward, and the sign reverses
   (interaction -0.142, p = 0.004).
3. **Representation, not information, is the cost of vision.** Rendering the
   same geometry as pixels for a CNN rather than a vector for an MLP costs
   0.16–0.24 success on every condition, and survives both 2.7× the compute
   and a reward that prices a crash like a stall — under which, unlike a
   coverage deficit, it still fails by stalling. An audit found the image
   carries depth to within 0.114 m, and the policy stopping in front of
   openings its own camera features mostly encode as open. A probe says what
   those features lack: the clearance ahead is there, the *width* of the gap
   past it is not (R² −0.29 against a depth vector's 0.22) — which is the
   quantity a robot needs to fit through a corridor.
4. **Nav2 corrected the baseline rather than confirming it.** The hand-written
   stack matches a production one to within 0.03 where clutter is not binding,
   but is 0.06–0.08 worse in tight corridors — so the published gap there was
   a lower bound. The mechanism was predicted in advance from failure-mode
   counts: replacing pure pursuit with a sampling local planner removed 58–92%
   of collisions, and only where the controller was the binding constraint.
5. **And it overturned this project's most favourable result.** Where the map
   is wrong, the learned policy had looked statistically indistinguishable
   from the planner. Against Nav2 that holds with sparse movers and fails once
   clutter is added, where Nav2 is 0.150–0.170 ahead of every training seed.
   A controlled subtraction then froze the movers in place — identical worlds,
   still absent from the map, only the motion removed — and the classical
   advantage returned and widened (p = 0.031, 0/6 seeds). The parity was
   motion degrading the planner, not the policy handling it. A third pass
   found the baseline a better replanning policy, which removed the last of
   the parity on cluttered dynamic worlds (−0.110, p = 0.031). All three came
   from deliberately re-testing the claim that most flattered the work.
6. **What motion costs the planner is a planning problem.**
   Given the movers' exact future positions, a planner that reasons in
   space-time and keeps a small temporal margin loses 0.010 to motion on dense
   clutter where the original lost 0.145. No robot has those positions:
   with the simplest real estimate, constant velocity from the robot's own
   observations, about half of that recovery survives on dense clutter and most
   of it on sparse. Four further changes to how the planner uses the estimate
   recovered none of the rest — but fitting each mover's oscillation from the
   robot's own observations recovered all of it (+0.070, p = 0.0001) and
   matched the oracle. The cost was the motion model, not the planner — but
   only with exact observations. A centimetre of error on each observed
   position costs more than the model ever bought (0.160 and 0.095 of success
   against 0.070), and at that accuracy the two estimators are
   indistinguishable. What a real stack needed there was neither: an estimator
   that does not amplify its own error. Fitting the same oscillation by least
   squares instead of differencing it is worth +0.145 on dense clutter at that
   noise, and leaves the planner indistinguishable from one handed the exact
   future — from the robot's own observations. Restricting those observations
   to what a sensor could see costs a 360° scanner little and a 90° camera
   nearly everything (−0.125), the report's coverage finding again.
7. **The reward decides what information is worth.** Frame stacking looked
   useless across three encodings and a 3× mover-speed range. It is not: the
   reward priced a collision at four times a timeout, and making the two equal
   turns stacking into a significant gain (+0.033, p = 0.019, 6 of 6 seeds)
   that is absent on a slow control. Stacking cuts collisions under either
   reward — the information was always used — but at 4:1 the saving becomes
   timeouts and at 1:1 it becomes successes. An observation channel is worth
   only what the objective lets the policy do with it.
8. **A causal test that failed usefully.** The explanation offered for the
   above — that replanning churn was the culprit — was pre-registered with a
   treated cell and a control. The treated cell moved exactly as predicted
   (+0.070) and the control moved by the same amount, which refutes the
   mechanism rather than confirming it. Without the control it would have read
   as a clean success and the report would assert something false.

## The part that matters most

**A correctly computed significance test produced a confident, reproducible,
and wrong conclusion.** An early result looked significant on one seed per arm
(+0.070, 95% CI [+0.006, +0.134]); across four seeds it vanished (+0.042,
p = 0.457). The interval was not miscalculated — it paired over *episodes*
when the unit of analysis is the *training seed*, and was silent on the
dominant source of variance. A correct answer to the wrong question is much
harder to notice than an error.

Everything afterwards uses seed-level analysis, exact permutation tests, and
pre-registered endpoints. Of thirty-three advance predictions, four derived from
*measurements* held; fourteen failed outright and fifteen got part right and part
wrong, with identical confidence of expression
throughout. The three failures that reasoned from a real measurement all
carried it into a regime nothing had been measured in — and the one that
crossed a boundary *with* the far side already measured held on both
its magnitude and its mechanism (+0.080,
collision interaction -0.133,
p = 0.024). Both the false positive and a
later harness bug that inverted a result are documented in the report rather
than quietly corrected. So is a third: the seed-comparison tool discarded the
timeout rate for the whole perception study, which turned one measured
behavioural shift into a published "changes nothing". Re-running all nine
comparisons with the repaired tool reproduced every number exactly and
recovered a result none of them could show alone — coverage deficits make the
robot get stuck, sensing deficits make it crash.

## What it demonstrates

Classical robotics (A\*, costmaps, pure pursuit, ROS 2 / Nav2) and modern
robot learning (PPO, domain randomisation, CNN encoders) in one controlled
comparison — plus the experimental discipline to catch a false positive in
one's own favour and publish it.

---

## Short forms

These are the project half of an application document. The personal half — why
this field, why this lab, where you want to go — is yours to write, and a
committee can tell the difference.

**~150 words, for a statement of purpose.**

> I built a controlled comparison between reinforcement-learning and
> search-based navigation, designed so the result could embarrass me. A PPO
> policy is scored against a classical A\* and pure-pursuit stack — and
> against real ROS 2 Nav2 — on identical procedurally generated worlds, under
> four distribution shifts, with the planner given the full map and exact
> pose. The planner wins everywhere, and I spent most of the project
> eliminating the comfortable explanations for that: insufficient data, wrong
> training distribution, insufficient compute, reward mis-specification. The
> cause turned out to be behavioural rather than any of them. Along the way a
> correctly computed confidence interval produced a confident and wrong
> conclusion, because it treated episodes rather than training seeds as the
> unit of analysis; I found it by replication, documented it in full, and
> reran every subsequent experiment with seed-level statistics and
> pre-registered endpoints.

**~60 words, for a CV entry or the opening of an email.**

> Vision + RL autonomous navigation (Python, PyTorch, Stable-Baselines3,
> ROS 2 / Nav2): a thirty-nine-experiment controlled study of learned versus
> classical navigation under distribution shift, with seed-level significance
> testing, pre-registered predictions, a real Nav2 baseline over a ROS 2
> bridge, and a 480-test suite. Includes a documented false positive I caught
> in my own results.

**One line, for a subject line or an introduction.**

> A controlled study of where learned navigation loses to classical planning —
> including the false positive I caught in my own results.
