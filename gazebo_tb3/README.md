# Virtual TurtleBot3

A TurtleBot3 in Gazebo Harmonic, driving the same procedurally generated worlds
as every other actor in the study. With no hardware available, this is the
transfer test that stands in for sim-to-real: everything measured in the 2-D
analytic simulator can be asked again of a physics-simulated robot with real
wheels, a real lidar model and wheel odometry.

## The robot

Nav2's own Gazebo TurtleBot3 Waffle (`nav2_minimal_tb3_sim`), pinned as
[`gz_waffle.sdf`](gz_waffle.sdf) and configured in code by
[`robot_sdf.py`](robot_sdf.py). Nav2's simulation is more generous than the
robot it models, so the default `real` profile applies the limits ROBOTIS
publishes for a TurtleBot3 Waffle Pi:

| | `real` (default) | `nav2` | 2-D simulator |
|---|---|---|---|
| top speed | 0.26 m/s | 0.46 m/s | 0.6 m/s |
| turn rate | 1.82 rad/s | 1.9 rad/s | 1.6 rad/s |
| lidar range | 0.12–3.5 m (LDS-01) | to 20 m | to 6 m |
| lidar rate | 5 Hz | 5 Hz | every step (10 Hz) |
| lidar position | 0.064 m behind the axle | same | at the centre |
| body | 0.265 m square behind the axle | same | 0.22 m disc |
| odometry | wheel encoders, physical slip | same | modelled noise |

The last four rows are kept as the real robot has them rather than idealised:
they are part of what a transfer test should face.

## The worlds

[`world_sdf.py`](world_sdf.py) rebuilds a world exactly: each circle a static
cylinder, each box a static box, the arena's edges four walls whose inner faces
sit on them, all 0.5 m tall so the lidar sees them as the 2-D sensor does.

## Running it

Gazebo runs headless and paused, and [`gz_session.py`](gz_session.py) steps it
exactly one control period at a time from Python, so a run is never limited by
Python's speed. World-control requests go through a small helper process: the
Python request call blocks while holding the interpreter lock, and in a process
that also receives sensor messages that deadlocks once the simulation is moving.
The first steps that need a lidar scan wait while the renderer starts, which
takes tens of seconds the first time under WSL.

The robot has its own RoboStack environment, `ros_tb3`, so the `ros_nav2`
environment the published Nav2 results were recorded in is untouched. In WSL:

```bash
micromamba create -n ros_tb3 -c robostack-jazzy -c conda-forge --strict-channel-priority \
  python=3.12 ros-jazzy-ros-base ros-jazzy-ros-gz-sim ros-jazzy-ros-gz-bridge \
  ros-jazzy-nav2-minimal-tb3-sim ros-jazzy-turtlebot3-msgs ros-jazzy-tf2-ros-py \
  numpy gymnasium omegaconf hydra-core pyyaml pytorch-cpu stable-baselines3
micromamba run -n ros_tb3 python gazebo_tb3/sensor_check.py --condition dense
```

## Checks

[`sensor_check.py`](sensor_check.py) sets the Gazebo lidar against the project's
own ray-cast from the same true sensor position, at the same bearings and range
limits, then drives the robot forward and sets its wheel odometry against the
true distance. Results for one `dense` and one `narrow` val world are in
[`results/gazebo_tb3/`](../results/gazebo_tb3/): the two sensors agree on which
beams hit something on more than 99% of beams, with a median range difference
under a centimetre, inside the lidar's own noise.

Known, and to be pinned before any experiment: Gazebo's lidar noise is not
seeded, and a velocity command can reach the drive on a slightly different
physics step, so repeated runs differ by millimetres.

## Files

| File | |
|---|---|
| `world_sdf.py` | A procedural world as a Gazebo SDF world. ROS-free, tested. |
| `robot_sdf.py` | The TurtleBot3 model, its profiles, and its lidar and drive specs. ROS-free, tested. |
| `gz_waffle.sdf` | Nav2's Waffle, expanded from xacro and pinned. |
| `expand_model.sh` | Regenerates `gz_waffle.sdf`. |
| `gz_session.py` | One Gazebo server, stepped in lockstep from Python. |
| `sensor_check.py` | The sensor and odometry agreement check. |
