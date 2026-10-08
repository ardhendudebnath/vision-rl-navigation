| Condition | Hand-written | Nav2 (2 runs) | Δ success | Reading |
|---|---|---|---|---|
| nominal | 1.000 / 0.985 | 0.970-0.980 / 0.955-0.963 | -0.030 to -0.020 | parity |
| dense | 0.890 / 0.841 | 0.910-0.940 / 0.881-0.901 | +0.020 to +0.050 | parity |
| sparse | 1.000 / 1.000 | 0.990-0.990 / 0.982-0.985 | -0.010 to -0.010 | parity |
| large | 1.000 / 0.990 | 0.990-0.990 / 0.983-0.985 | -0.010 to -0.010 | parity |
| narrow | 0.850 / 0.795 | 0.910-0.930 / 0.878-0.898 | +0.060 to +0.080 | Nav2 better |
| noisy_lidar | 1.000 / 0.985 | 0.980-0.990 / 0.966-0.976 ‡ | -0.020 to -0.010 | parity |

‡ Corrected by hand from `results/nav2_noise_rerun/run*_rtf5/`. The script that
writes this table reads `results/nav2_runs/`, whose `noisy_lidar` passes had
Nav2 on a clean sensor (0.970-0.980 / 0.953-0.964); see report §9.20.

## Moving obstacles

Two passes each, same protocol. The learned column is the best arm over
its six training seeds.

| Condition | Hand-written (succ / coll) | Learned (best) | Nav2 (succ / coll) | Nav2 − learned |
|---|---|---|---|---|
| dynamic | 0.870 / 0.130 | 0.860 ± 0.033 | 0.840–0.870 / 0.130–0.160 | -0.020 to +0.010 |
| **dynamic_dense** | 0.750 / 0.250 | 0.710 ± 0.042 | 0.860–0.880 / 0.110–0.140 | **+0.150 to +0.170** |

Parity with the learned policy survives on `dynamic` and fails on
`dynamic_dense`, where Nav2 is above all six training seeds. Nav2 gets
there by removing clutter collisions (0.250 → 0.110–0.140), not by
handling movers — on `dynamic` its collision rate (0.130–0.160) is no
better than the hand-written baseline's 0.130. See report §9.3.
