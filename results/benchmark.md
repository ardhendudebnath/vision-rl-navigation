# Benchmark results

Every row is evaluated on the same worlds in the same order.
`nominal` is the held-out in-distribution test split; the remaining
conditions are distribution shifts never seen during training.

| Condition | Actor | Success | SPL | Collision | Timeout | Steps (succ.) |
|---|---|---|---|---|---|---|
| nominal | random | 0.000 | 0.000 | 0.980 | 0.020 | 0 |
| nominal | classical | 1.000 | 0.985 | 0.000 | 0.000 | 161 |
| nominal | nominal_trained | 0.960 | 0.910 | 0.040 | 0.000 | 152 |
| nominal | dr_trained | 0.940 | 0.865 | 0.030 | 0.030 | 150 |
| nominal | dr_long | 0.930 | 0.885 | 0.020 | 0.050 | 142 |
| dense | random | 0.000 | 0.000 | 1.000 | 0.000 | 0 |
| dense | classical | 0.890 | 0.841 | 0.090 | 0.020 | 226 |
| dense | nominal_trained | 0.640 | 0.593 | 0.290 | 0.070 | 176 |
| dense | dr_trained | 0.660 | 0.593 | 0.120 | 0.220 | 175 |
| dense | dr_long | 0.680 | 0.631 | 0.080 | 0.240 | 173 |
| sparse | random | 0.010 | 0.009 | 0.900 | 0.090 | 258 |
| sparse | classical | 1.000 | 1.000 | 0.000 | 0.000 | 143 |
| sparse | nominal_trained | 0.980 | 0.963 | 0.010 | 0.010 | 142 |
| sparse | dr_trained | 0.990 | 0.951 | 0.010 | 0.000 | 139 |
| sparse | dr_long | 0.970 | 0.950 | 0.020 | 0.010 | 135 |
| large | random | 0.000 | 0.000 | 0.920 | 0.080 | 0 |
| large | classical | 1.000 | 0.990 | 0.000 | 0.000 | 215 |
| large | nominal_trained | 0.970 | 0.940 | 0.010 | 0.020 | 210 |
| large | dr_trained | 0.970 | 0.925 | 0.000 | 0.030 | 205 |
| large | dr_long | 0.970 | 0.928 | 0.010 | 0.020 | 207 |
| narrow | random | 0.000 | 0.000 | 1.000 | 0.000 | 0 |
| narrow | classical | 0.850 | 0.795 | 0.120 | 0.030 | 224 |
| narrow | nominal_trained | 0.600 | 0.556 | 0.270 | 0.130 | 178 |
| narrow | dr_trained | 0.630 | 0.563 | 0.110 | 0.260 | 178 |
| narrow | dr_long | 0.640 | 0.585 | 0.020 | 0.340 | 167 |
| noisy_lidar | random | 0.000 | 0.000 | 0.980 | 0.020 | 0 |
| noisy_lidar | classical | 1.000 | 0.985 | 0.000 | 0.000 | 161 |
| noisy_lidar | nominal_trained | 0.960 | 0.911 | 0.040 | 0.000 | 151 |
| noisy_lidar | dr_trained | 0.960 | 0.884 | 0.030 | 0.010 | 151 |
| noisy_lidar | dr_long | 0.940 | 0.896 | 0.020 | 0.040 | 143 |

## Reading this table

- `random` is the floor. Any result that does not clearly clear it is noise.
- **`noisy_lidar` is a no-op for `classical` by construction.** The classical
  baseline navigates from the map and never reads the lidar, so its row is
  identical to `nominal`. That is not robustness — it is non-exposure, and it
  is exactly the axis on which a sensor-driven policy should be expected to
  differ.
- SPL can reach 1.000 when the agent stops inside the goal tolerance and so
  travels slightly less than `l*`; the `max(p, l*)` term caps it there.

_Shift definitions: dense, large, narrow, nominal, sparse._