| Clutter | Actor | Frozen | Moving | Cost of motion |
|---|---|---|---|---|
| sparse | classical | 0.980 | 0.870 | **-0.110** |
|  | learned (dyn4) | 0.900 | 0.852 | **-0.048** |
|  | Nav2 | 0.985 | 0.855 | **-0.130** |
| dense | classical | 0.910 | 0.750 | **-0.160** |
|  | learned (dyn4) | 0.778 | 0.710 | **-0.068** |
|  | Nav2 | 0.945 | 0.870 | **-0.075** |

- sparse: Nav2 leads the hand-written planner by +0.005 with the movers frozen and -0.015 with them moving, so little of its advantage is motion handling rather than clutter handling.
- dense: Nav2 leads the hand-written planner by +0.035 with the movers frozen and +0.120 with them moving, so most of its advantage is motion handling rather than clutter handling.
