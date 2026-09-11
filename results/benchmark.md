# Benchmark results

Every row is evaluated on the same worlds in the same order.
`nominal` is the held-out in-distribution test split; the remaining
conditions are distribution shifts never seen during training.

| Condition | Actor | Success | SPL | Collision | Timeout | Steps (succ.) |
|---|---|---|---|---|---|---|
| nominal | random | 0.000 | 0.000 | 0.980 | 0.020 | 0 |
| nominal | classical | 1.000 | 0.985 | 0.000 | 0.000 | 161 |
| dense | random | 0.000 | 0.000 | 1.000 | 0.000 | 0 |
| dense | classical | 0.890 | 0.841 | 0.090 | 0.020 | 226 |
| sparse | random | 0.010 | 0.009 | 0.900 | 0.090 | 258 |
| sparse | classical | 1.000 | 1.000 | 0.000 | 0.000 | 143 |
| large | random | 0.000 | 0.000 | 0.920 | 0.080 | 0 |
| large | classical | 1.000 | 0.990 | 0.000 | 0.000 | 215 |
| narrow | random | 0.000 | 0.000 | 1.000 | 0.000 | 0 |
| narrow | classical | 0.850 | 0.795 | 0.120 | 0.030 | 224 |
| noisy_lidar | random | 0.000 | 0.000 | 0.980 | 0.020 | 0 |
| noisy_lidar | classical | 1.000 | 0.985 | 0.000 | 0.000 | 161 |

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