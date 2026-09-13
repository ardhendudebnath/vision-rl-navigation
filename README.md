# Learning Vision-Conditioned Navigation Policies

### A comparative study against classical planning

![Classical planner vs learned policy, side by side](results/demo_comparison.gif)

*Left: A\* + pure pursuit with the full map. Right: PPO from a 64-beam lidar.
Same world, same clock. Six held-out episodes across open, cluttered and
tight-corridor worlds. Grey: the global plan (classical only — the learned
policy has no plan to draw). Orange: the executed trajectory.
[Full-quality MP4](results/demo_comparison.mp4).*

**The episodes are stratified to match the measured outcome rates, not
hand-picked wins.** Taking the first six seeds in order gave six successes for
both actors, which misrepresents a policy measured at 0.70 success on
`narrow`. In this clip the learned policy succeeds in 4 of 6 (0.67, against
0.70–0.73 measured) and the classical planner in 5 of 6 (0.83, against
0.85–0.89) — including one world where **the classical planner is the one that
crashes**. Every seed is listed in
[`scripts/make_comparison_video.py`](scripts/make_comparison_video.py) so the
selection is reproducible.

---

A reinforcement-learning navigation agent, a classical planning stack, and a
protocol strict enough that comparing them means something.

The question is not "can an RL agent reach a goal" — it is **where a learned
policy beats a strong classical planner, where it loses, and how each one
degrades when the world stops looking like the training set.** Every design
decision below follows from wanting that comparison to be trustworthy.

## Status

| Stage | State |
|---|---|
| Task definition, world generation, metrics, test suite | Done |
| Classical baseline (A* + pure pursuit, full map access) | Done |
| Privileged RL (PPO on pose + ranges) | Done — 1.5M steps, val SPL 0.894 |
| Robustness suite across shifted environments | Done |
| Domain randomisation over shifts | Done — did not close the gap |
| Compute sweep to 4.0M steps | Done — rejects the compute explanation |
| Caution-vs-progress reward ablation | Done — rejects the reward explanation |
| Lidar beam-count experiment | Done — looked decisive on one seed |
| Multi-seed replication (4 seeds × 2 arms) | Done — **the beam result does not replicate** |
| Direct perception audit (no training, no seeds) | Done — deficit is real but small; explains the null |
| 16 vs 64 beams, 6 seeds/arm, pre-registered | Done — **significant; perception confirmed** |
| Depth camera vs lidar, 6 seeds/arm, pre-registered | Done — **field of view beats resolution** |
| FOV sweep (90/180/270/360°), 24 seeds, pre-registered | Done — **monotone trend, forecast held** |
| Decoupling FOV from sample count | Done — **coverage causal, samples inert** |
| RGB + CNN encoder, 6 seeds/arm, pre-registered | Done — **the pixels are the problem** |
| RGB compute sweep to 4.0M | Done — gap survives 2.7x compute |
| Moving obstacles absent from the map | Done — **gap shrinks 8x, does not reverse** |
| Trained on movers + frame stacking | Done — **parity, not victory; stacking inert** |
| Technical report + demo video | Done |

**Technical report: [`docs/report.md`](docs/report.md)** — the full study written
up as a short paper, including the false positive this project caught in its
own results and how. Phase-by-phase detail and rationale:
[`docs/project_plan.md`](docs/project_plan.md).

## Results

100 held-out worlds per condition. All actors see **identical worlds in
identical order**. `nominal` is the in-distribution held-out split; the rest
are distribution shifts. Success rate / SPL:

| Condition | Classical | PPO nominal 1.5M | PPO DR 1.5M | PPO DR 4.0M |
|---|---|---|---|---|
| nominal | **1.000** / 0.985 | 0.960 / 0.910 | 0.940 / 0.865 | 0.930 / 0.885 |
| sparse | **1.000** / 1.000 | 0.980 / 0.963 | 0.990 / 0.951 | 0.970 / 0.950 |
| large | **1.000** / 0.990 | 0.970 / 0.940 | 0.970 / 0.925 | 0.970 / 0.928 |
| noisy_lidar | 1.000 / 0.985 * | 0.960 / 0.911 | 0.960 / 0.884 | 0.940 / 0.896 |
| dense | **0.890** / 0.841 | 0.640 / 0.593 | 0.660 / 0.593 | 0.680 / 0.631 |
| narrow | **0.850** / 0.795 | 0.600 / 0.556 | 0.630 / 0.563 | 0.640 / 0.585 |

\* `noisy_lidar` is a no-op for the classical planner *by construction* — it
navigates from the map and never reads the lidar. Non-exposure, not
robustness. Random scores 0.000 everywhere except `sparse` (0.010).

Full table: [`results/benchmark.md`](results/benchmark.md).

### The finding: the policy learns not to crash, not how to get through

A strong classical planner beat every learned policy on every condition. The
interesting part is *why*, and it is not visible in the success rate.

The project's hypothesis was that a reactive policy, not committed to a
precomputed path, would close the planner's cornering weakness under clutter.
That weakness was measured precisely: under shift **A\* never once fails to
find a route** — 0 planning failures, with failures being the controller
losing the plan while cornering. When RL lost instead, the obvious suspect was
out-of-distribution brittleness, so a second policy was trained on worlds
randomised over arena size, obstacle count, obstacle size and start-goal
separation — ranges chosen to *contain* every evaluation shift — then given
2.7x the compute.

Success rate barely moved. The failure *composition* moved enormously:

**`narrow`, 100 episodes:**

| Policy | Successes | Collisions | Timeouts | Timeout progress |
|---|---|---|---|---|
| nominal, 1.5M | 60 | 27 | 13 | 4.4 m of 10.6 m, 0.09 m/s |
| DR, 1.5M | 63 | 11 | 26 | 7.5 m of 11.7 m, 0.15 m/s |
| DR, 4.0M | **64** | **2** | **34** | 7.1 m of 11.8 m, 0.14 m/s |

Collisions fall 27 → 11 → **2**: the policy has very nearly learned to stop
crashing. But of the 25 episodes that left the collision bucket, **21 became
timeouts and only 4 became successes.** The stalled episodes crawl at
0.14 m/s against a 0.6 m/s cap and run out of budget two-thirds of the way
there; the classical planner finishes `dense` in 226 steps.

More training, and a wider training distribution, both buy *safety* and
neither buys *completion*. The policy converges on caution.

### Caveats, including one that corrects an earlier claim

- **Correction: the DR policy was not compute-limited.** An earlier version of
  this README said it was "still improving at 1.5M" and discounted the result
  on that basis. That read was two points of a noisy 50-episode validation
  curve. Extending to 4.0M shows it plateaus by ~1.75M and then drifts
  slightly down (best val SPL 0.770, ending at 0.651). The compute explanation
  is now tested and rejected, which makes the caution finding stronger, not
  weaker.
- **The comparison is structurally asymmetric.** Both policies train on a
  distribution; the classical planner has none, so on shifted conditions this
  is an in-distribution planner against an out-of-distribution policy. Nor is
  it equal-compute: DR got 2.7x more. Both asymmetries favour the learned
  side, and it still lost.
- **The DR ranges were chosen to contain the evaluation shifts,** so this
  tests whether widening the training distribution recovers the loss, *not*
  generalisation to unseen kinds of shift. Evaluation worlds remain held-out
  seeds, so this is not leakage in the memorisation sense, but the
  distribution is deliberately matched.
- **n = 100 per condition**, so success-rate differences under ~0.05 are not
  resolvable. The collision/timeout shifts above are far larger than that; the
  success-rate changes are not.

### What the learned policies do win

Both small, both caveated: fewer steps on successful episodes (142 vs 161 in
`nominal` for the 4.0M DR policy, ~12% faster, mildly flattered by
survivorship), and indifference to 0.10 m lidar range noise — the one axis the
classical baseline cannot be compared on at all, since it never reads the
sensor.

### The stalling is the reward's stated preference, not a training failure

Measuring actual episode returns for the 4.0M policy settles *why* it stalls:

| Outcome | n | Mean return |
|---|---|---|
| success | 64 | **+41.20** |
| timeout | 34 | **−2.06** |
| collision | 2 | **−24.91** |

A collision costs a flat −20; timing out for all 500 steps costs −5
(`step_penalty` 0.01 × 500). **Crashing is four times worse than stalling
forever**, so the policy is not malfunctioning — it found the optimum of the
reward it was given.

### Retuning the caution does not buy successes — it buys collisions

Three arms, each isolating one caution term, all trained from scratch at 1.5M
on the DR distribution. `narrow`, 100 held-out episodes:

| Policy | Success | Collisions | Timeouts |
|---|---|---|---|
| classical | **0.850** | 0.120 | 0.030 |
| DR baseline | 0.630 | 0.110 | 0.260 |
| `abl_noprox` (`proximity_penalty` 0.15→0) | 0.660 | 0.100 | 0.240 |
| `abl_step` (`step_penalty` 0.01→0.05) | 0.660 | 0.280 | 0.060 |
| `abl_lowcoll` (`collision_penalty` 20→5) | 0.560 | **0.440** | **0.000** |

As caution falls, timeouts convert into collisions almost one-for-one while
**success stays pinned in a 0.56–0.66 band**. Paired tests over the identical
worlds (n=100) confirm it: no arm significantly improves success rate under
clutter, and `abl_lowcoll` significantly *hurts* `nominal` success
(−0.070, 95% CI [−0.127, −0.013]).

So the caution terms control **which** failure you get, not **how many**. The
reward balance is not the bottleneck.

### One real, significant win

`abl_step` improves `nominal` SPL by **+0.066 (95% CI [+0.019, +0.113])** — the
one significant gain in the whole ablation. Success is unchanged, so this is
not about caution at all: the baseline was *dawdling even when it succeeded*,
and charging more per step cleans up the paths. Worth keeping; it does not
touch the clutter problem.

### Perception is the bottleneck — established, after two false starts

This took four phases and one retracted claim, so the short version first:

1. **32 vs 64 beams, one seed each** — looked significant. It wasn't; see below.
2. **32 vs 64 beams, four seeds each** — null (+0.042, p = 0.457). Seed spread
   swamped the effect.
3. **A training-free audit of the sensor itself** — the perception deficit is
   real and quantified, but 32 → 64 recovers only 6.4 points of gap detection,
   far too little to see through that noise. It predicted that **16 vs 64**
   (a 17-point contrast) would work.
4. **16 vs 64 beams, six seeds each** — **significant, and the audit's
   predicted effect size held.**

The final result, on the pre-registered primary endpoint (`narrow` success
rate, seed as the unit of analysis, exact permutation test):

| Arm | Per-seed success | Mean ± sd |
|---|---|---|
| 16 beams | 0.67, 0.51, 0.60, 0.59, 0.58, 0.63 | 0.597 ± 0.054 |
| 64 beams | 0.70, 0.58, 0.64, 0.74, 0.71, 0.72 | **0.682 ± 0.060** |

**+0.085, p = 0.035.** On `dense` the separation is complete — every 64-beam
seed above every 16-beam seed — giving **+0.115, p = 0.002**, the floor for
6-vs-6, along with SPL +0.083 (p = 0.004) and collisions −0.087 (p = 0.039).
`nominal` does not reach significance (+0.038, p = 0.056).

The audit predicted roughly +0.11 from the detection contrast before any of
these policies were trained. Observed: +0.085 to +0.115. **A training-free
measurement forecast the outcome of a twelve-run training experiment**, which
is stronger evidence for the mechanism than the effect size alone.

Everything below documents how the two false starts happened, because that is
the more useful part.

### Field of view beats angular resolution

With the sensor established as a real constraint, the natural question is
*which* property of it matters. A forward-facing depth camera and a 360°
lidar trade off in opposite directions:

| Sensor | Angular resolution | Resolves a 0.44 m gap to | World visible |
|---|---|---|---|
| 64 beams / 360° | 5.62° | 4.5 m | 100% |
| 64 columns / 90° | **1.41°** | **17.9 m** | **25%** |

The camera is **4× finer per degree while seeing a quarter of the world**.
Both give a 69-dimensional observation to an identical network, so capacity is
matched and the sensor is the only difference. Six seeds per arm, `narrow`
success pre-registered as the primary endpoint.

**The camera loses, on every condition:**

| Condition | Lidar 360° | Depth 90° | Δ | p (exact) |
|---|---|---|---|---|
| **narrow** (primary) | 0.682 ± 0.060 | 0.585 ± 0.037 | **−0.097** | **0.019** |
| dense | 0.695 ± 0.040 | 0.565 ± 0.053 | **−0.130** | **0.004** |
| nominal | 0.937 ± 0.021 | 0.862 ± 0.039 | **−0.075** | **0.004** |

So **field of view dominates angular resolution** at this operating point.
Quadrupling angular precision does not come close to paying for losing three
quarters of the view — consistent with Phase 2h, which found 64-beam gap
detection already at 0.949 and therefore with little left for extra resolution
to buy.

The penalty also **grows with clutter** — −0.075 on `nominal`, −0.097 on
`narrow`, −0.130 on `dense` — which is what peripheral awareness being the
scarce resource looks like: the more obstacles there are, the more it costs
not to see beside and behind you.

Collision rates are statistically unchanged on `narrow` and `dense`
(+0.012, +0.008, both n.s.). The camera policy does not crash more there; it
**times out** more (0.298 vs 0.213 on `narrow`), which is the same
cautious-rather-than-capable failure the lidar policies showed under clutter.

This outcome was **predicted in advance** and recorded before the runs
started. A confirmed prediction is weaker evidence than a surprising one —
it cannot rule out that the reasoning was fitted to an expected answer — but
it does mean the Phase 2h saturation argument had real forecasting content.

### The FOV curve, and a second forecast that held

Sweeping field of view at a **fixed 64 samples** turns that direction into a
curve. Widening the view necessarily coarsens it, and 360°/64 columns *is* the
64-beam lidar — so the sweep interpolates continuously between the two arms
above. The audit ran first, before any of the new policies existed:

| FOV | Spacing | Gap detection | Predicted success | **Observed** |
|---|---|---|---|---|
| 90° | 1.41° | 0.401 | 0.585 *(anchor)* | 0.585 ± 0.037 |
| 180° | 2.81° | 0.564 | **0.614** | **0.635 ± 0.087** |
| 270° | 4.22° | 0.718 | **0.641** | **0.642 ± 0.066** |
| 360° | 5.62° | 0.949 | 0.682 *(anchor)* | 0.682 ± 0.060 |

**Spearman ρ = +0.508, p = 0.013** over 24 seeds. No adjacent pair is
individually significant — exactly as the audit predicted, which is why a
monotone trend was pre-registered as the test instead of pairwise comparisons.

The two interior levels were genuine out-of-sample predictions and came in
within **+0.021** and **+0.001**. (The tooling prints a mean absolute error of
0.006 across all four levels; that figure flatters the forecast, since two
points were the anchors used to fit the slope. The honest out-of-sample figure
is 0.011.)

This is the second training-free forecast of a training result in this project.
Two independent quantitative predictions from the same audit is the strongest
evidence here that the mechanism is real rather than a story fitted afterwards.

**One guess that was wrong:** mid-sweep, validation SPL suggested the curve
saturated at 180–270°. It doesn't — held-out success keeps rising to 360°.
Validation SPL is a max-over-checkpoints statistic on the training
distribution, the same one that misled in the replication above.

### Decoupled: coverage is causal, sample count is inert

All four levels above use 64 samples, so coverage and resolution move together
by construction. Since resolution = FOV/samples they cannot both be pinned —
so the follow-up pins **resolution** at 2.81°/sample and varies coverage.
Three comparisons, six seeds each, all pre-registered, with the two predicted
nulls given **explicit magnitude bounds** rather than the unfalsifiable "not
significant":

| Comparison | Holds fixed | Predicted | **Observed** | p |
|---|---|---|---|---|
| 64@90° vs 32@90° | FOV | null, <0.01 | **+0.002** | 1.000 |
| 128@360° vs 64@360° | FOV | null, <0.01 | **−0.003** | 0.955 |
| **128@360° vs 32@90°** | **2.81°/sample** | **+0.10** | **+0.095** | **0.024** |

**Doubling the sample count changes nothing — twice, at both ends of the FOV
range. Quadrupling coverage at identical angular resolution produces the whole
effect.** (`dense` agrees: +0.088, p = 0.028.)

This sharpens the earlier framing rather than confirming it. "Field of view
beats angular resolution" implied a frontier where either knob buys
performance. There is no such frontier here: **one knob is inert.**

It also closes a loose end — a single-seed run had suggested 128 beams was
*worse* than 64. Across six seeds the difference is −0.003. Seed noise, like
the 32-vs-64 case before it.

*One secondary, recorded but not claimed:* at 360°, the 128-beam arm collides
more than the 64-beam arm (0.188 vs 0.105, p = 0.006) with success unchanged.
Across ~9 secondary tests the Bonferroni threshold is 0.0056, so this **does
not survive correction.**

### Where the map is wrong, the gap nearly closes — but does not reverse

Every condition above hands the classical planner a **perfect, current, static
map** — its largest privilege, and the one real deployments don't have. The
final experiment adds obstacles that **move and are absent from the map**.

**Hypothesis, recorded beforehand: the learned policy would win here. It does
not.** Success rate, 100 worlds, 6 seeds per arm, classical at its best
configuration on each condition:

| Condition | Classical | Lidar 360° | Δ | sign test |
|---|---|---|---|---|
| nominal | **1.000** | 0.937 ± 0.021 | −0.063 | 0/6, p = 0.031 |
| narrow | **0.850** | 0.682 ± 0.060 | **−0.168** | 0/6, p = 0.031 |
| dynamic_dense | **0.750** | 0.700 ± 0.023 | −0.050 | 0/6, p = 0.031 |
| **dynamic** | **0.870** | 0.848 ± 0.019 | **−0.022** | 0/6, **p = 0.062** |

No seed beats the baseline anywhere. But the gap **shrinks about eightfold**
between tight static corridors and moving obstacles, and `dynamic` is the only
condition in the whole study where classical's advantage **isn't
statistically established**. Reactive control closes most of the distance
exactly where the map degrades — it just doesn't overtake.

**A near-miss worth recording.** I added replanning to make the baseline
stronger under movers. It does help there (+0.06 to +0.07) — but it *hurts*
on static conditions, dropping `narrow` from 0.850 to 0.690 through path churn
in tight corridors. Using one global setting, `narrow` would have shown
classical at 0.690 against the policy's 0.682, with 4/6 seeds above the
baseline: a clean "parity in tight corridors" claim that was purely an
artifact of a handicap I'd introduced in the name of fairness. The comparison
now gives the classical planner its best configuration per condition.

### Parity where the map is wrong — but motion information is inert

The zero-shot policies above structurally *could not* anticipate movers: a
single scan gives obstacle positions, not velocities, so a one-frame policy is
in the same position as the replanning planner. Frame stacking fixes that —
two scans encode motion — so the follow-up trained on movers with a 4-frame
stack, against a 1-frame control.

**Predicted: the stacked policy beats classical by +0.01 to +0.05. It does
not.**

| Condition | Classical | `dyn1` (1 frame) | `dyn4` (4 frames) |
|---|---|---|---|
| dynamic | 0.870 | 0.860 ± 0.033 | 0.852 ± 0.038 |
| dynamic_dense | 0.750 | 0.697 ± 0.060 | 0.710 ± 0.042 |

**Frame stacking did nothing**: −0.008 (p = 0.784) and +0.013 (p = 0.703)
against its control. The one structural advantage the learned side had
produced no measurable benefit. Training on movers at all was also negligible
(+0.012 vs zero-shot, p = 0.541).

What *is* real is a regime change:

| Regime | Δ vs classical | Significant? |
|---|---|---|
| static clutter (`narrow`) | −0.168 | **yes**, p = 0.031 |
| moving obstacles | −0.010 | no, p = 0.625 |
| moving obstacles, cluttered | −0.053 | no, p = 0.219 |

**Parity, not victory.** Where the map is wrong the learned policy becomes
statistically indistinguishable from a strong replanning planner; where the
map is right it stays clearly worse. Across the whole study that is the
closest RL comes to winning.

### The pixels are the problem, not the geometry

Every result above uses range data. The final experiment uses images, built so
that **only the representation changes**: the RGB camera renders the same
geometry the depth camera measures, at the same 90° FOV and the same 64
columns, with the goal vector bit-identical between modes. It isolates what it
costs to make a CNN recover from pixels what an MLP reads directly.

| Condition | Depth (MLP) | RGB (CNN) | Δ | p |
|---|---|---|---|---|
| **narrow** (primary) | 0.585 ± 0.037 | 0.367 ± 0.090 | **−0.218** | **0.002** |
| dense | 0.565 ± 0.053 | 0.327 ± 0.097 | **−0.238** | **0.002** |
| nominal | 0.862 ± 0.039 | 0.700 ± 0.059 | **−0.162** | **0.002** |

All three at the permutation floor: **every depth seed beats every RGB seed on
every condition.** The encoder also costs *reliability* — seed spread roughly
doubles for success and quadruples for collisions (individual RGB seeds range
from 0.02 to 0.45 collision rate).

**My prediction was wrong.** I forecast −0.05 to −0.10; the effect is two to
three times that. Worth contrasting with the two forecasts that held (above):
those were derived from a quantity that had actually been *measured*. This one
was an intuition in the same confident register, and it was badly calibrated.

**The compute excuse, tested and rejected.** The CNN has far more parameters
at the same budget, so the RGB policies were resumed to **4.0M steps — 2.7×
the depth arm's**:

| Comparison | Δ | p |
|---|---|---|
| rgb@4.0M vs rgb@1.5M | +0.057 | 0.524 (n.s.) |
| **rgb@4.0M vs depth@1.5M** | **−0.162** | **0.030** |

Extra compute doesn't significantly help, and **the gap survives handing RGB
2.7× the compute of the arm it loses to.** Compute accounts for about a
quarter of the original 0.218; the rest stays. Seed variance actually *rose*
with more training (±0.153 vs ±0.090) — the opposite of convergence.

The render is still clean — no texture, lighting, or sensor noise — so this
remains a *lower bound* on the encoder's cost.

### The false start, and why it fooled a correct significance test

`narrow`, success rate, one value per training seed (100 held-out worlds each):

| Arm | Per-seed success | Mean ± sd |
|---|---|---|
| 32 beams | 0.63, 0.68, 0.66, 0.52 | 0.623 ± 0.071 |
| 64 beams | 0.70, 0.58, 0.64, 0.74 | 0.665 ± 0.070 |

Exact permutation test with the **seed** as the unit of analysis:
**+0.042, p = 0.457.** Not significant. The same holds on `dense`
(+0.073, p = 0.229) and `nominal` (+0.007, p = 0.800).

**The seed-to-seed spread (±0.07) is larger than the effect (+0.042).** The
two arms' ranges overlap almost completely: 0.52–0.68 against 0.58–0.74.

Worse for the original claim, the collision result was pure seed luck. Phase
2f reported `narrow` collisions falling 0.110 → 0.030. Across seeds, the two
arms are **identical**: 0.077 ± 0.025 versus 0.080 ± 0.048. Seed 0 of the
32-beam arm happened to be its *worst* for collisions and seed 0 of the
64-beam arm its *best*, and the single-seed comparison picked up exactly that.

What survives is weak and honest: the direction is positive in all six
condition×metric comparisons, and best-validation SPL separated completely
across seeds (every 64-beam seed above every 32-beam seed, 0.787 ± 0.009 vs
0.743 ± 0.014). But those six deltas are not independent, and "best validation
SPL" is a *maximum* over ~30 checkpoints, which both inflates it and
suppresses its variance. The held-out benchmark is the honest measure, and it
does not resolve an effect at n = 4.

**So: 64 beams may help a little; this experiment cannot show that it does.**
The geometric argument below still predicts an effect and is still worth
testing — it simply needs more seeds, or a larger beam contrast, than four
runs per arm can settle.

### The lesson, which applies to every result above

Phases 2c–2f each compared **single training runs** with significance measured
by pairing over *episodes*. Pairing over episodes controls world difficulty
and is the right test for "do these two policies differ on these worlds" — but
it is silent on training-seed variance, and here that variance turned out to
be larger than every effect being reported. Treat the single-seed findings
above (including the `abl_step` SPL win) as unreplicated.

The one conclusion that is robust to all of this: **the classical planner beats
every learned policy on every condition**, by margins far larger than the seed
spread (0.850 vs 0.665 on `narrow`).

### The geometric argument that motivated the beam experiment

Compute, training distribution and reward balance were all eliminated. The
remaining suspect was the sensor, and the arithmetic is blunt:

| Beams | Angular spacing | Resolves a robot-width gap out to |
|---|---|---|
| 32 | 11.25° | **2.24 m** |
| 64 | 5.62° | 4.48 m |
| 128 | 2.81° | 8.96 m |

With 32 beams the robot could not reliably see a gap it would fit through
beyond **2.24 m** — barely more than a body length of lookahead. Doubling to
64 beams pushes that to 4.48 m.

On a single seed per arm, retraining with 64 beams appeared to give the first
significant success-rate improvement under clutter: `narrow` success +0.070,
paired over episodes, 95% CI [+0.006, +0.134]. **That is the result the
multi-seed replication above dissolves**, and it is left here as the record of
what a single-seed comparison looked like when it was wrong.

128 beams was no better than the 32-beam baseline anywhere. With the seed
analysis in hand, that non-monotonicity is best read as seed noise too, not as
evidence about observation dimensionality.

### Measuring the mechanism directly — which reconciles both results

Rather than keep inferring perception limits from success rate, the limit is
now measured head-on: across 480 on-route poses in 60 `narrow` worlds, how
often does an N-beam scan fail to reveal an opening the robot could actually
drive through? No policy, no training, no seeds
([`scripts/perception_audit.py`](scripts/perception_audit.py)).

| Beams | Spacing | Traversable gaps detected |
|---|---|---|
| 16 | 22.50° | 0.778 |
| 32 | 11.25° | 0.885 |
| 64 | 5.62° | 0.949 |
| 128 | 2.81° | 0.974 |

Detection rate by how wide the gap appears:

| Gap width | n | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| 0–5° | 132 | 0.129 | **0.258** | 0.553 | 0.773 |
| 5–10° | 122 | 0.451 | 0.738 | 1.000 | 1.000 |
| 10–20° | 203 | 0.635 | 0.980 | 1.000 | 1.000 |
| 20–45° | 372 | 0.995 | 1.000 | 1.000 | 1.000 |
| >45° | 335 | 1.000 | 1.000 | 1.000 | 1.000 |

**The perception deficit is real, and it is exactly where the geometry said it
would be**: a 32-beam scan misses 11.5% of traversable gaps overall, and
**74% of the gaps narrower than 5°**. Every gap wider than 20° is seen by
every sensor.

**And this is why the end-to-end experiment came out null.** Going 32 → 64
beams recovers only **6.4 percentage points** of gap detection (0.885 →
0.949). An effect that small has no chance of clearing a ±0.07 seed spread in
success rate at four seeds per arm. Phase 2f was not wrong to look here — it
was underpowered, and the audit quantifies by how much rather than leaving it
a guess.

It also says what a better experiment looks like: **16 vs 64 beams spans
0.778 → 0.949, a 17-point gap in detection — nearly three times the 32 → 64
contrast.** That is the manipulation most likely to produce a success-rate
effect that survives seed noise, and the audit identified it without training
a single policy.

### The powered experiment, and how well the prediction held

Twelve runs, six seeds per arm, primary endpoint declared before looking at
the data. Success rate:

| Condition | 16 beams | 64 beams | Δ | p (exact) |
|---|---|---|---|---|
| **narrow** (primary) | 0.597 ± 0.054 | 0.682 ± 0.060 | **+0.085** | **0.035** |
| dense | 0.580 ± 0.035 | 0.695 ± 0.040 | **+0.115** | **0.002** |
| nominal | 0.898 ± 0.041 | 0.937 ± 0.021 | +0.038 | 0.056 |

On `dense` the arms separate completely — every 64-beam seed above every
16-beam seed — which is why p sits at the 2/924 floor. SPL there is +0.083
(p = 0.004) and collisions −0.087 (p = 0.039).

**On multiplicity:** `narrow` success was pre-registered as the single primary
endpoint, so its p = 0.035 needs no correction — that is what pre-registration
buys. The eight secondary tests do need it: at Bonferroni (0.05/8 = 0.006),
`dense` success and `dense` SPL survive and the rest do not. Stated plainly
because p = 0.035 would *not* survive correction if `narrow` were treated as
one of nine exploratory tests, and the only thing separating those two
readings is having fixed the endpoint in advance.

The effect also scales with detection roughly as the audit implies: a 6.4-point
detection contrast gave +0.042 (not significant), a 17-point contrast gives
+0.085 — a 2.7× manipulation producing a 2.0× effect.

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/activate   # Linux/macOS: source .venv/bin/activate
pip install -e ".[dev,viz]"
pytest
```

Evaluate the classical baseline on held-out worlds:

```bash
python -m vision_nav.training.evaluate --actor classical --split test --episodes 100
```

Check the training pipeline end to end (about two minutes, CPU):

```bash
python -m vision_nav.training.train --config-name smoke
```

Train the privileged-RL agent properly:

```bash
python -m vision_nav.training.train
```

Train with domain randomisation, or continue an existing run:

```bash
python -m vision_nav.training.train env=nav_dr train.run_name=ppo_dr
```

```bash
python -m vision_nav.training.train env=nav_dr train.run_name=ppo_dr_long train.total_timesteps=2500000 train.resume_from=runs/ppo_dr/final_model.zip
```

Run the full comparison matrix and write the results table:

```bash
python scripts/run_benchmark.py --rl nominal=runs/ppo_privileged/best_model.zip dr=runs/ppo_dr/best_model.zip
```

Render the side-by-side comparison video (MP4 + GIF):

```bash
python scripts/make_comparison_video.py --rl runs/beams64/best_model.zip --out results/demo_comparison
```

Render a single-actor GIF:

```bash
python scripts/make_demo.py --world 20000 --worlds 3 --out results/demo.gif
```

## The task

A differential-drive robot must reach a goal pose in a procedurally generated
12x12 m arena of circular and box obstacles, using a 32-beam planar lidar and
a goal vector. Worlds are generated from an integer seed and are **guaranteed
solvable** — start and goal are collision-free, at least 5 m apart, and
verified connected before the episode begins.

That guarantee is load-bearing. If some episodes were impossible, a failure
would be ambiguous between "bad policy" and "bad task", and success rate and
SPL would both stop meaning anything.

**Observation** (37-d): 32 normalised lidar ranges, goal distance, goal
bearing as `(cos, sin)`, current linear and angular velocity.
**Action** (2-d, continuous): normalised `(v, omega)` — deliberately the same
interface as `geometry_msgs/Twist`, so a real ROS 2 base is a transport change
rather than a policy rewrite.

## Metrics

Success rate, **SPL**, collision rate, timeout rate, steps to goal, and path
efficiency. SPL follows Anderson et al. (2018), *On Evaluation of Embodied
Navigation Agents*, so the numbers are comparable to published work rather
than being project-specific scores.

Two details that turned out to matter more than expected:

- **`l*` is string-pulled before use.** A raw 8-connected A* path
  overestimates the true geodesic distance, and an overestimated `l*` makes
  `l* / max(p, l*)` clamp to 1.0 for any competent policy — SPL quietly stops
  discriminating between good and great. Line-of-sight shortcutting fixes it,
  and the effect is measurable: baseline path efficiency went from exactly
  1.000 (saturated, useless) to 0.993 (real).
- **Mean steps-to-goal averages successes only.** Including failures would let
  a policy look fast by crashing early.

## Design decisions worth arguing about

**Reward shaping is geodesic, not Euclidean.** Progress reward uses an A*
distance field. Euclidean shaping creates a local optimum behind every
obstacle: the agent gets paid to press into a wall that happens to lie between
it and the goal. The distance field is *training-time* privileged information
— it never enters the observation, so the policy remains honestly sensor-only
at evaluation. `reward.use_geodesic_progress=false` keeps the ablation
available.

**Model selection uses validation SPL, not training return.** Return is shaped
and not comparable across configurations. Selecting on the metric the report
actually presents avoids picking a checkpoint that learned to farm the shaping
term instead of navigating.

**The classical baseline is given every advantage.** Full obstacle map, exact
pose. A baseline that loses because it was handicapped proves nothing.

## Repository layout

```
src/vision_nav/
  envs/        World generation, robot kinematics, lidar, the Gymnasium task
  planning/    A*, geodesic distance fields, path smoothing
  agents/      Classical A* + pure-pursuit baseline
  metrics/     Success rate, SPL, collisions, path efficiency
  training/    PPO training, evaluation harness, the shared actor interface
  viz/         Top-down renderer for figures and demo videos
configs/       Hydra configs (env / algo / train)
scripts/       Benchmark matrix, demo rendering
tests/         75 tests over geometry, planning, env contract and metrics
docs/          Project plan, decisions and status
```

## Hardware

Developed on a laptop RTX 5070 Ti (12 GB VRAM), which is **below** Isaac Sim's
documented 16 GB minimum. The simulator here is deliberately lightweight
(pure NumPy, analytic geometry, ~2,000 env steps/s on CPU) so the full
pipeline can be built and debugged locally, with Isaac Lab / Habitat runs
reserved for rented GPU time. The task definition, metrics, splits and
evaluation harness all carry over unchanged when the simulator swaps.

## License

MIT — see [LICENSE](LICENSE).
