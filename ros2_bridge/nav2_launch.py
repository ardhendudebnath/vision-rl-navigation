"""Minimal Nav2 bringup for the simulator bridge.

Why not ``nav2_bringup/navigation_launch.py``: that launch starts eleven
lifecycle nodes and wires the command chain as

    controller_server -> /cmd_vel_nav -> velocity_smoother
                      -> /cmd_vel_smoothed -> collision_monitor -> /cmd_vel

Every node in that list must configure successfully or the lifecycle manager
aborts the *whole* bringup, and the ones this study does not need
(collision_monitor, route_server, docking_server, waypoint_follower) each
demand their own parameter block. Worse, if collision_monitor is absent the
final hop is missing and no velocity command ever reaches the simulator.

So this launch starts exactly the four nodes point-to-point navigation needs
and lets the controller publish straight to ``/cmd_vel``:

    controller_server -> /cmd_vel -> bridge

The nodes left out only ever *slow* the robot (velocity_smoother re-limits
acceleration the env already limits; collision_monitor is a safety governor),
so omitting them can only help Nav2. That is the right direction: a baseline
that loses through handicap proves nothing.

    ros2 launch ros2_bridge/nav2_launch.py params_file:=ros2_bridge/nav2_params.yaml
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

#: Brought up in this order by the lifecycle manager. behavior_server must be
#: configured before bt_navigator, which looks up its action servers.
LIFECYCLE_NODES = [
    "controller_server",
    "planner_server",
    "behavior_server",
    "bt_navigator",
]


def generate_launch_description() -> LaunchDescription:
    params_file = LaunchConfiguration("params_file")
    use_sim_time = LaunchConfiguration("use_sim_time")

    common = {
        "parameters": [params_file, {"use_sim_time": use_sim_time}],
        "output": "screen",
        # The stock launch remaps these so nodes use the relative topic; keep
        # that, otherwise tf2 listeners in each node miss the broadcasts.
        "remappings": [("/tf", "tf"), ("/tf_static", "tf_static")],
    }

    nodes = [
        Node(package="nav2_controller", executable="controller_server",
             name="controller_server", **common),
        Node(package="nav2_planner", executable="planner_server",
             name="planner_server", **common),
        Node(package="nav2_behaviors", executable="behavior_server",
             name="behavior_server", **common),
        Node(package="nav2_bt_navigator", executable="bt_navigator",
             name="bt_navigator", **common),
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_navigation",
            output="screen",
            parameters=[
                {"use_sim_time": use_sim_time,
                 "autostart": True,
                 "node_names": LIFECYCLE_NODES}
            ],
        ),
    ]

    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file"),
            DeclareLaunchArgument("use_sim_time", default_value="True"),
            *nodes,
        ]
    )
