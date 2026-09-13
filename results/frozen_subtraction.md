| Clutter | Actor | Frozen | Moving | Cost of motion |
|---|---|---|---|---|
| sparse | classical | 1.000 | 0.880 | **-0.120** |
|  | learned (dyn4) | 0.900 | 0.852 | **-0.048** |
|  | Nav2 | 0.985 | 0.855 | **-0.130** |
| dense | classical | 0.980 | 0.820 | **-0.160** |
|  | learned (dyn4) | 0.778 | 0.710 | **-0.068** |
|  | Nav2 | 0.945 | 0.870 | **-0.075** |

- sparse: Nav2 stands -0.015 against the hand-written planner with the movers frozen and -0.025 with them moving — a swing of -0.010 attributable to motion alone, which is inside the noise band.
- dense: Nav2 stands -0.035 against the hand-written planner with the movers frozen and +0.050 with them moving — a swing of +0.085 attributable to motion alone, which is outside the noise band.
