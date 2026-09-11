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
| nominal | abl_noprox | 0.920 | 0.895 | 0.070 | 0.010 | 144 |
| nominal | abl_lowcoll | 0.870 | 0.821 | 0.130 | 0.000 | 144 |
| dense | random | 0.000 | 0.000 | 1.000 | 0.000 | 0 |
| dense | classical | 0.890 | 0.841 | 0.090 | 0.020 | 226 |
| dense | dr_base | 0.660 | 0.593 | 0.120 | 0.220 | 175 |
| dense | abl_step | 0.600 | 0.573 | 0.340 | 0.060 | 165 |
| dense | abl_noprox | 0.610 | 0.574 | 0.250 | 0.140 | 172 |
| dense | abl_lowcoll | 0.540 | 0.508 | 0.460 | 0.000 | 149 |
| sparse | random | 0.010 | 0.009 | 0.900 | 0.090 | 258 |
| sparse | classical | 1.000 | 1.000 | 0.000 | 0.000 | 143 |
| sparse | dr_base | 0.990 | 0.951 | 0.010 | 0.000 | 139 |
| sparse | abl_step | 0.990 | 0.980 | 0.010 | 0.000 | 140 |
| sparse | abl_noprox | 0.970 | 0.951 | 0.020 | 0.010 | 138 |
| sparse | abl_lowcoll | 0.960 | 0.923 | 0.040 | 0.000 | 138 |
| large | random | 0.000 | 0.000 | 0.920 | 0.080 | 0 |
| large | classical | 1.000 | 0.990 | 0.000 | 0.000 | 215 |
| large | dr_base | 0.970 | 0.925 | 0.000 | 0.030 | 205 |
| large | abl_step | 0.970 | 0.950 | 0.010 | 0.020 | 208 |
| large | abl_noprox | 0.960 | 0.939 | 0.020 | 0.020 | 203 |
| large | abl_lowcoll | 0.990 | 0.949 | 0.010 | 0.000 | 204 |
| narrow | random | 0.000 | 0.000 | 1.000 | 0.000 | 0 |
| narrow | classical | 0.850 | 0.795 | 0.120 | 0.030 | 224 |
| narrow | dr_base | 0.630 | 0.563 | 0.110 | 0.260 | 178 |
| narrow | abl_step | 0.660 | 0.623 | 0.280 | 0.060 | 169 |
| narrow | abl_noprox | 0.660 | 0.621 | 0.100 | 0.240 | 172 |
| narrow | abl_lowcoll | 0.560 | 0.500 | 0.440 | 0.000 | 170 |
| noisy_lidar | random | 0.000 | 0.000 | 0.980 | 0.020 | 0 |
| noisy_lidar | classical | 1.000 | 0.985 | 0.000 | 0.000 | 161 |
| noisy_lidar | dr_base | 0.940 | 0.868 | 0.040 | 0.020 | 150 |
| noisy_lidar | abl_step | 0.940 | 0.922 | 0.050 | 0.010 | 142 |
| noisy_lidar | abl_noprox | 0.910 | 0.885 | 0.080 | 0.010 | 145 |
| noisy_lidar | abl_lowcoll | 0.910 | 0.859 | 0.090 | 0.000 | 144 |

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