"""ROS 2 bridge exposing ProceduralNavEnv to Nav2.

The report's largest standing caveat is that the classical baseline is
*structurally analogous* to Nav2 rather than being Nav2. This node closes it:
the same worlds, the same seed order, the same action space and the same
metrics, with the real Nav2 stack driving instead of the hand-written
A* + pure-pursuit controller.

What Nav2 is given
------------------
Deliberately generous, matching how the hand-written baseline is treated
throughout the project — a baseline that loses through handicap proves
nothing:

- the **static** occupancy grid as ``/map`` (movers stay absent from it,
  exactly as in the dynamic-obstacle experiments),
- ground-truth pose via TF and ``/odom`` — no localisation error,
- a **360-beam** ``/scan``, denser than anything the learned policies get,
  so the local costmap is not the bottleneck.

Taking those privileges away
----------------------------
Report §9.2 and §9.3 took the map and then the pose away from the hand-written
stack and measured what each was worth. ``privileges="slam"`` does the same
here, so the two answers are comparable: the bridge stops publishing ``/map``
and stops publishing ``map -> odom``, leaving both to ``slam_toolbox``, and
``odom -> base_link`` carries a *drifting* estimate instead of the truth.

The drift comes from :class:`~vision_nav.mapping.localisation.DeadReckoning`
with the default :class:`OdometryConfig` — the same model, the same parameters
and the same per-world seeding as §9.3, imported rather than reimplemented.
A second odometry model would have made the difference between the two stacks
partly a difference between two noise generators.

Time
----
The simulator is a 10 Hz stepped process, not a wall-clock one. The node
publishes ``/clock`` and every Nav2 node runs with ``use_sim_time``, so the
planner's notion of time advances exactly one control period per sim step.
Without this, Nav2's costmap update rates and controller frequencies would be
racing a simulator that only moves when told to.
"""

from __future__ import annotations

import math
import os
from dataclasses import replace

import numpy as np
import rclpy
from geometry_msgs.msg import Quaternion, TransformStamped, Twist, TwistStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from vision_nav.envs import NavEnvConfig, ProceduralNavEnv
from vision_nav.envs.sensors import Lidar2D
from vision_nav.mapping.localisation import DeadReckoning, OdometryConfig

#: Beams in the scan handed to Nav2. Far denser than the learned policies get
#: (16-128), so the local costmap is never the limiting factor.
NAV2_SCAN_BEAMS = 360

#: Message type on ``/cmd_vel``. Must match ``enable_stamped_cmd_vel`` in
#: nav2_params.yaml. Jazzy defaults to stamped; Humble publishes plain Twist.
CMD_VEL_TYPE = TwistStamped

LATCHED = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)


def yaw_to_quaternion(yaw: float) -> Quaternion:
    return Quaternion(z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))


class Nav2Bridge(Node):
    """Publishes the simulator to ROS 2 and applies Nav2's velocity commands."""

    def __init__(self, env_config: NavEnvConfig, privileges: str = "full",
                 beams: int = NAV2_SCAN_BEAMS) -> None:
        super().__init__("nav2_bridge")
        if privileges not in ("full", "slam"):
            raise ValueError(f"privileges must be 'full' or 'slam', got {privileges!r}")
        #: ``"full"``: the static map and the exact pose. ``"slam"``: neither —
        #: slam_toolbox owns ``/map`` and ``map -> odom``, and this node
        #: publishes a drifting ``odom -> base_link``.
        self.privileges = privileges
        self.slam = privileges == "slam"
        self.env = ProceduralNavEnv(env_config)
        self.dt = env_config.robot.dt
        # §9.3's odometry, imported so the two stacks drift the same way.
        # NAV2_PERFECT_ODOM=1 turns the noise off: a development control, which
        # separates what SLAM does from what the odometry feeding it does.
        noisy = self.slam and not os.environ.get("NAV2_PERFECT_ODOM")
        self._odom = DeadReckoning(env_config.robot, OdometryConfig() if noisy else None)
        # Only the beam count is replaced. Range, FOV and — critically — the
        # noise and dropout of the evaluation condition are inherited from the
        # env config: building a fresh LidarConfig here would hand Nav2 a clean
        # sensor under `noisy_lidar` and quietly score the wrong experiment.
        #
        # 360 by default, as in every published Nav2 row. The SLAM comparison
        # also runs at 32, the scanner the hand-written mapping stack of report
        # §9.2 built its map from: at 360 against 32, a difference between the
        # two stacks would be partly a sensor ten times denser.
        self.beams = int(beams)
        self._scan = Lidar2D(replace(env_config.lidar, n_beams=self.beams))

        self._cmd = np.zeros(2)  # (v, omega) in SI units, as Nav2 sends them
        self._sim_time = 0.0
        self._episode_active = False
        self.last_info: dict = {}
        self.outcome: str | None = None
        self.steps = 0
        self.commands_received = 0
        self._map_msg: OccupancyGrid | None = None

        self.clock_pub = self.create_publisher(Clock, "/clock", 10)
        self.scan_pub = self.create_publisher(LaserScan, "/scan", 10)
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.map_pub = self.create_publisher(OccupancyGrid, "/map", LATCHED)
        self.tf = TransformBroadcaster(self)
        self.static_tf = StaticTransformBroadcaster(self)

        # Exactly one type: the RMW refuses two subscriptions of different
        # types on the same topic ("incompatible type ... rt/cmd_vel"), so the
        # bridge cannot hedge. Jazzy's controller_server and behavior_server
        # publish TwistStamped when enable_stamped_cmd_vel is true, which
        # nav2_params.yaml sets explicitly for both. If that ever disagrees
        # with CMD_VEL_TYPE the robot simply never moves, so run_nav2_eval
        # checks commands_received and fails loudly rather than reporting a
        # sweep of timeouts as Nav2's performance.
        if CMD_VEL_TYPE is TwistStamped:
            self.create_subscription(TwistStamped, "/cmd_vel",
                                     self._on_twist_stamped, 10)
        else:
            self.create_subscription(Twist, "/cmd_vel", self._on_twist, 10)

        self._publish_static_tf()

    # ------------------------------------------------------------------
    def _on_twist(self, msg: Twist) -> None:
        self._cmd = np.array([msg.linear.x, msg.angular.z])
        self.commands_received += 1

    def _on_twist_stamped(self, msg: TwistStamped) -> None:
        self._cmd = np.array([msg.twist.linear.x, msg.twist.angular.z])
        self.commands_received += 1

    def _publish_static_tf(self) -> None:
        """base_link -> laser, identity: the scanner sits at the robot origin."""
        t = TransformStamped()
        t.header.frame_id = "base_link"
        t.child_frame_id = "laser"
        t.transform.rotation.w = 1.0
        self.static_tf.sendTransform(t)

    # ------------------------------------------------------------------
    def now_msg(self):
        sec = int(self._sim_time)
        return sec, int((self._sim_time - sec) * 1e9)

    def _stamp(self, header) -> None:
        sec, nsec = self.now_msg()
        header.stamp.sec = sec
        header.stamp.nanosec = nsec

    def start_episode(self, world_seed: int) -> dict:
        """Reset the world for a new episode.

        Sim time deliberately keeps running across episodes instead of
        restarting at zero. Nav2 treats a backwards clock as a time jump and
        purges its TF buffers and costmap history, so a per-episode reset
        would leave every episode after the first navigating from a corrupted
        state.
        """
        _, info = self.env.reset(options={"world_seed": world_seed})
        self._cmd[:] = 0.0
        self.steps = 0
        self.outcome = None
        self._episode_active = True
        self.last_info = info
        self._map_msg = None  # new layout; rebuild on next publish
        # Seeded from the world, exactly as §9.3 seeds it, so a world drifts
        # the same way in both stacks. Starting from the true pose is not a
        # privilege: it is the origin the map frame is defined by.
        self._odom.rng = np.random.default_rng((int(world_seed), 7))
        self._odom.reset(self.env.robot.pose)
        self.publish_all()
        self.publish_map()
        return info

    @property
    def believed_pose(self) -> np.ndarray:
        """Where odometry alone says the robot is. In ``slam`` mode this is the
        ``odom`` frame's answer, before slam_toolbox's correction."""
        return self._odom.pose

    def action_from_cmd(self) -> np.ndarray:
        """Map Nav2's (v, omega) in SI units into the env's normalised action.

        Delegates to the robot model rather than reimplementing the mapping:
        an inverse that disagreed with ``scale_action`` would systematically
        distort every velocity Nav2 commands, handicapping the baseline in a
        way no amount of staring at the metrics would reveal.
        """
        return self.env.robot.unscale_action(self._cmd).astype(np.float32)

    def step_sim(self) -> bool:
        """Advance one control period. Returns False when the episode ends."""
        if not self._episode_active:
            return False
        _, _, terminated, truncated, info = self.env.step(self.action_from_cmd())
        # The encoders read the velocity the base actually achieved, which is
        # what the robot integrates. With privileges="full" the model is
        # noiseless and this reproduces the true pose exactly.
        self._odom.update(self.env.robot.velocity)
        self.steps += 1
        self._sim_time += self.dt
        self.last_info = info
        if terminated or truncated:
            self._episode_active = False
            self.outcome = (
                "success" if info.get("is_success")
                else "collision" if info.get("collision")
                else "timeout"
            )
            return False
        return True

    # ------------------------------------------------------------------
    def publish_clock(self) -> None:
        msg = Clock()
        sec, nsec = self.now_msg()
        msg.clock.sec = sec
        msg.clock.nanosec = nsec
        self.clock_pub.publish(msg)

    def publish_all(self) -> None:
        self.publish_clock()
        self.publish_tf()
        self.publish_odom()
        self.publish_scan()

    def publish_tf(self) -> None:
        """``map -> odom -> base_link``.

        With full privileges, ``map -> odom`` is identity and ``odom ->
        base_link`` is the ground-truth pose — the same privilege the
        hand-written baseline receives, so localisation error is not a confound.

        In ``slam`` mode only ``odom -> base_link`` is published here, carrying
        the drifting estimate, because ``map -> odom`` is precisely the
        correction slam_toolbox exists to compute. Publishing both would leave
        two broadcasters fighting over one edge, and TF would take whichever
        arrived last.
        """
        x, y, yaw = self.believed_pose if self.slam else self.env.robot.pose
        edges = [("odom", "base_link", float(x), float(y), yaw_to_quaternion(float(yaw)))]
        if not self.slam:
            edges.insert(0, ("map", "odom", 0.0, 0.0, yaw_to_quaternion(0.0)))
        for parent, child, tx, ty, rot in edges:
            t = TransformStamped()
            self._stamp(t.header)
            t.header.frame_id = parent
            t.child_frame_id = child
            t.transform.translation.x = tx
            t.transform.translation.y = ty
            t.transform.rotation = rot
            self.tf.sendTransform(t)

    def publish_odom(self) -> None:
        x, y, yaw = self.believed_pose if self.slam else self.env.robot.pose
        v, omega = self.env.robot.velocity
        msg = Odometry()
        self._stamp(msg.header)
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_link"
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.orientation = yaw_to_quaternion(float(yaw))
        msg.twist.twist.linear.x = float(v)
        msg.twist.twist.angular.z = float(omega)
        self.odom_pub.publish(msg)

    def publish_scan(self) -> None:
        ranges = self._scan.scan(self.env.world, self.env.robot.pose)
        msg = LaserScan()
        self._stamp(msg.header)
        msg.header.frame_id = "laser"
        msg.angle_min = -math.pi
        msg.angle_max = math.pi - (2 * math.pi / self.beams)
        msg.angle_increment = 2 * math.pi / self.beams
        msg.range_min = 0.0
        msg.range_max = float(self._scan.config.max_range)
        msg.scan_time = self.dt
        msg.time_increment = 0.0
        msg.ranges = [float(r) for r in ranges]
        self.scan_pub.publish(msg)

    def publish_map(self) -> None:
        """The STATIC occupancy grid, uninflated.

        Uninflated because Nav2's costmap applies its own inflation layer;
        publishing a pre-inflated grid would double-count the robot radius and
        make the planner refuse gaps it could actually drive through.

        Cached per episode: the grid is thousands of clearance queries and this
        is called on every pump of the startup loop.

        Nothing is published in ``slam`` mode: the map is what the robot is
        there to build, and slam_toolbox owns the topic.
        """
        if self.slam:
            return
        if self._map_msg is not None:
            self._stamp(self._map_msg.header)
            self.map_pub.publish(self._map_msg)
            return

        world = self.env.world
        res = world.config.grid_resolution
        # Cell-centre occupancy of the static layout only.
        n_x = int(np.ceil(world.config.width / res))
        n_y = int(np.ceil(world.config.height / res))
        xs = (np.arange(n_x) + 0.5) * res
        ys = (np.arange(n_y) + 0.5) * res
        gx, gy = np.meshgrid(xs, ys, indexing="xy")
        pts = np.stack([gx.ravel(), gy.ravel()], axis=-1)
        blocked = world.clearance(pts, include_dynamic=False) <= 0.0

        msg = OccupancyGrid()
        self._stamp(msg.header)
        msg.header.frame_id = "map"
        msg.info.resolution = res
        msg.info.width = n_x
        msg.info.height = n_y
        msg.info.origin.orientation.w = 1.0
        msg.data = [100 if b else 0 for b in blocked.astype(bool)]
        self._map_msg = msg
        self.map_pub.publish(msg)


def main() -> None:  # pragma: no cover - entry point
    rclpy.init()
    node = Nav2Bridge(NavEnvConfig())
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":  # pragma: no cover
    main()
