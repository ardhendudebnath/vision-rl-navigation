# Benchmark results

Every row is evaluated on the same worlds in the same order.
`nominal` is the held-out in-distribution test split; the remaining
conditions are distribution shifts never seen during training.

| Condition | Actor | Success | SPL | Collision | Timeout | Steps (succ.) |
|---|---|---|---|---|---|---|
| nominal | random | 0.000 | 0.000 | 0.980 | 0.020 | 0 |
| nominal | classical | 1.000 | 0.985 | 0.000 | 0.000 | 161 |
| nominal | dr_base | 0.940 | 0.865 | 0.030 | 0.030 | 150 |
| nominal | abl_step | 0.950 | 0.931 | 0.040 | 0.010 | 143 |
| nominal | beams64 | 0.960 | 0.905 | 0.030 | 0.010 | 146 |
| nominal | beams128 | 0.930 | 0.877 | 0.040 | 0.030 | 153 |
| dense | random | 0.000 | 0.000 | 1.000 | 0.000 | 0 |
| dense | classical | 0.890 | 0.841 | 0.090 | 0.020 | 226 |
| dense | dr_base | 0.660 | 0.593 | 0.120 | 0.220 | 175 |
| dense | abl_step | 0.600 | 0.573 | 0.340 | 0.060 | 165 |
| dense | beams64 | 0.730 | 0.680 | 0.110 | 0.160 | 172 |
| dense | beams128 | 0.630 | 0.571 | 0.250 | 0.120 | 187 |
| sparse | random | 0.010 | 0.009 | 0.900 | 0.090 | 258 |
| sparse | classical | 1.000 | 1.000 | 0.000 | 0.000 | 143 |
| sparse | dr_base | 0.990 | 0.951 | 0.010 | 0.000 | 139 |
| sparse | abl_step | 0.990 | 0.980 | 0.010 | 0.000 | 140 |
| sparse | beams64 | 1.000 | 0.970 | 0.000 | 0.000 | 138 |
| sparse | beams128 | 0.980 | 0.945 | 0.000 | 0.020 | 143 |
| large | random | 0.000 | 0.000 | 0.920 | 0.080 | 0 |
| large | classical | 1.000 | 0.990 | 0.000 | 0.000 | 215 |
| large | dr_base | 0.970 | 0.925 | 0.000 | 0.030 | 205 |
| large | abl_step | 0.970 | 0.950 | 0.010 | 0.020 | 208 |
| large | beams64 | 0.990 | 0.951 | 0.000 | 0.010 | 204 |
| large | beams128 | 0.980 | 0.934 | 0.010 | 0.010 | 212 |
| narrow | random | 0.000 | 0.000 | 1.000 | 0.000 | 0 |
| narrow | classical | 0.850 | 0.795 | 0.120 | 0.030 | 224 |
| narrow | dr_base | 0.630 | 0.563 | 0.110 | 0.260 | 178 |
| narrow | abl_step | 0.660 | 0.623 | 0.280 | 0.060 | 169 |
| narrow | beams64 | 0.700 | 0.629 | 0.030 | 0.270 | 181 |
| narrow | beams128 | 0.650 | 0.576 | 0.220 | 0.130 | 199 |
| noisy_lidar | random | 0.000 | 0.000 | 0.980 | 0.020 | 0 |
| noisy_lidar | classical | 1.000 | 0.985 | 0.000 | 0.000 | 161 |
| noisy_lidar | dr_base | 0.940 | 0.870 | 0.040 | 0.020 | 149 |
| noisy_lidar | abl_step | 0.950 | 0.931 | 0.040 | 0.010 | 144 |
| noisy_lidar | beams64 | 0.970 | 0.910 | 0.030 | 0.000 | 147 |
| noisy_lidar | beams128 | 0.940 | 0.883 | 0.030 | 0.030 | 154 |

## Reading this table

- `random` is the floor. Any result that does not clearly clear it is noise.
- **Actors here use different lidar beam counts** ([32, 64, 128]).
  Sensor geometry is read from each run's saved config, because a policy
  cannot be evaluated at a beam count it was not trained for. The worlds,
  seed order and injected sensor noise are identical across every row —
  asserted at evaluation time, not assumed.
- **`noisy_lidar` is a no-op for `classical` by construction.** The classical
  baseline navigates from the map and never reads the lidar, so its row is
  identical to `nominal`. That is not robustness — it is non-exposure, and it
  is exactly the axis on which a sensor-driven policy should be expected to
  differ.
- SPL can reach 1.000 when the agent stops inside the goal tolerance and so
  travels slightly less than `l*`; the `max(p, l*)` term caps it there.

_Shift definitions: dense, large, narrow, nominal, sparse._