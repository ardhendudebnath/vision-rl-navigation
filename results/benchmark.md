# Benchmark results

Every row is evaluated on the same worlds in the same order.
`nominal` is the held-out in-distribution test split; the remaining
conditions are distribution shifts never seen during training.

| Condition | Actor | Success | SPL | Collision | Timeout | Steps (succ.) |
|---|---|---|---|---|---|---|
| nominal | random | 0.000 | 0.000 | 0.980 | 0.020 | 0 |
| nominal | classical | 1.000 | 0.985 | 0.000 | 0.000 | 161 |
| nominal | lidar64 | 0.960 | 0.905 | 0.030 | 0.010 | 146 |
| nominal | depth64 | 0.880 | 0.839 | 0.100 | 0.020 | 147 |
| dense | random | 0.000 | 0.000 | 1.000 | 0.000 | 0 |
| dense | classical | 0.890 | 0.841 | 0.090 | 0.020 | 226 |
| dense | lidar64 | 0.730 | 0.680 | 0.110 | 0.160 | 172 |
| dense | depth64 | 0.590 | 0.549 | 0.160 | 0.250 | 172 |
| sparse | random | 0.010 | 0.009 | 0.900 | 0.090 | 258 |
| sparse | classical | 1.000 | 1.000 | 0.000 | 0.000 | 143 |
| sparse | lidar64 | 1.000 | 0.970 | 0.000 | 0.000 | 138 |
| sparse | depth64 | 0.950 | 0.914 | 0.050 | 0.000 | 140 |
| large | random | 0.000 | 0.000 | 0.920 | 0.080 | 0 |
| large | classical | 1.000 | 0.990 | 0.000 | 0.000 | 215 |
| large | lidar64 | 0.990 | 0.951 | 0.000 | 0.010 | 204 |
| large | depth64 | 0.950 | 0.895 | 0.020 | 0.030 | 215 |
| narrow | random | 0.000 | 0.000 | 1.000 | 0.000 | 0 |
| narrow | classical | 0.850 | 0.795 | 0.120 | 0.030 | 224 |
| narrow | lidar64 | 0.700 | 0.629 | 0.030 | 0.270 | 181 |
| narrow | depth64 | 0.600 | 0.561 | 0.100 | 0.300 | 166 |
| noisy_lidar | random | 0.000 | 0.000 | 0.980 | 0.020 | 0 |
| noisy_lidar | classical | 1.000 | 0.985 | 0.000 | 0.000 | 161 |
| noisy_lidar | lidar64 | 0.960 | 0.904 | 0.040 | 0.000 | 147 |
| noisy_lidar | depth64 | 0.910 | 0.865 | 0.070 | 0.020 | 148 |

## Reading this table

- `random` is the floor. Any result that does not clearly clear it is noise.
- **Actors here use different sensors:**
  - `depth64` — depth 64px @ 90deg
  - `lidar64` — lidar 64 beams @ 360deg
  Each policy is evaluated with the sensor it was trained on, read from
  its saved config, because a policy cannot be run on a sensor it has
  never seen. The worlds, seed order and injected sensor noise are
  identical across every row — asserted at evaluation time, not assumed.
  Sensor noise is applied to whichever sensor the policy actually reads,
  so the `noisy_lidar` row is a genuine perturbation for the depth
  policies too, not the non-exposure it is for `classical`.
- **`noisy_lidar` is a no-op for `classical` by construction.** The classical
  baseline navigates from the map and never reads the lidar, so its row is
  identical to `nominal`. That is not robustness — it is non-exposure, and it
  is exactly the axis on which a sensor-driven policy should be expected to
  differ.
- SPL can reach 1.000 when the agent stops inside the goal tolerance and so
  travels slightly less than `l*`; the `max(p, l*)` term caps it there.

_Shift definitions: dense, large, narrow, nominal, sparse._