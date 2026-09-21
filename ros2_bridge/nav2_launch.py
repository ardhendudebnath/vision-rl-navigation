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
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler, TimerAction
from launch.conditions import IfCondition
from launch.events import matches_action
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition

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
    slam = LaunchConfiguration("slam")
    slam_params_file = LaunchConfiguration("slam_params_file")

    common = {
        "parameters": [params_file, {"use_sim_time": use_sim_time}],
        "output": "screen",
        # The stock launch remaps these so nodes use the relative topic; keep
        # that, otherwise tf2 listeners in each node miss the broadcasts.
        "remappings": [("/tf", "tf"), ("/tf_static", "tf_static")],
    }

    # Nav2's global costmap refuses to activate until the `map` frame exists,
    # and gives up half a second after asking. With full privileges the bridge
    # publishes map -> odom from the first message, so there is nothing to wait
    # for. Under SLAM that frame does not exist until slam_toolbox has received
    # a scan and processed it, which lost the race every time: the costmap
    # aborted, the lifecycle manager aborted the whole bringup, and nothing ran.
    # So navigation is started after SLAM here, which is also the order a real
    # robot brings them up in.
    navigation = [
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

    # The SLAM arm (report §9.4), started only when asked for. It publishes /map
    # and the map -> odom correction that the bridge stops publishing.
    #
    # slam_toolbox 2.8 is a lifecycle node and does not configure itself. The
    # first version of this launch started it as a plain Node on the belief that
    # it would, and it sat unconfigured for the whole run — no scan processed,
    # no map frame — while Nav2's global costmap waited for a frame that could
    # never appear. It is driven through configure and activate here exactly as
    # slam_toolbox's own online_sync_launch.py drives it, and kept out of
    # LIFECYCLE_NODES so the navigation manager cannot abort over it.
    slam_node = LifecycleNode(
        package="slam_toolbox",
        executable="sync_slam_toolbox_node",
        name="slam_toolbox",
        namespace="",
        output="screen",
        parameters=[slam_params_file,
                    {"use_sim_time": use_sim_time, "use_lifecycle_manager": False}],
        remappings=[("/tf", "tf"), ("/tf_static", "tf_static")],
        condition=IfCondition(slam),
    )
    slam_configure = EmitEvent(
        event=ChangeState(lifecycle_node_matcher=matches_action(slam_node),
                          transition_id=Transition.TRANSITION_CONFIGURE),
        condition=IfCondition(slam),
    )
    slam_activate = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=slam_node,
            start_state="configuring",
            goal_state="inactive",
            entities=[EmitEvent(event=ChangeState(
                lifecycle_node_matcher=matches_action(slam_node),
                transition_id=Transition.TRANSITION_ACTIVATE))],
        ),
        condition=IfCondition(slam),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file"),
            DeclareLaunchArgument("use_sim_time", default_value="True"),
            DeclareLaunchArgument("slam", default_value="False"),
            DeclareLaunchArgument("slam_params_file", default_value=""),
            # Wall-clock seconds before navigation starts. 0 with full
            # privileges, where there is nothing to wait for; run_nav2.sh sets
            # it for the SLAM arm.
            DeclareLaunchArgument("nav2_delay", default_value="0.0"),
            slam_node,
            slam_activate,
            slam_configure,
            TimerAction(period=LaunchConfiguration("nav2_delay"), actions=navigation),
        ]
    )
