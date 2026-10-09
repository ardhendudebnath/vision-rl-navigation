"""The virtual TurtleBot3: Nav2's Gazebo Waffle, set to a real Waffle Pi's limits.

The model is Nav2's own (``nav2_minimal_tb3_sim``, ``gz_waffle.sdf.xacro``,
expanded with xacro): the Waffle's body, wheels and casters, a 360-sample GPU
lidar at 5 Hz with 0.01 m of range noise, and Gazebo's differential-drive
plugin publishing wheel odometry. Nav2's simulation is more generous than the
robot it models -- 0.46 m/s, a 20 m lidar -- so the ``real`` profile sets the
limits ROBOTIS publishes for the TurtleBot3 Waffle Pi: 0.26 m/s, 1.82 rad/s,
and the LDS-01's 0.12 to 3.5 m. The ``nav2`` profile leaves Nav2's values.

Two properties of the real robot are kept rather than idealised, because they
are part of what a transfer test should face: the lidar sits 0.064 m behind
the point the robot turns about, and the body is a 0.265 m square offset
behind the wheel axis rather than a disc around it.

Kept free of ROS and Gazebo imports so it can be tested anywhere.
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET

#: Published limits, ROBOTIS e-Manual, TurtleBot3 Waffle Pi and LDS-01.
PROFILES = {
    "real": {"max_linear_velocity": 0.26, "max_angular_velocity": 1.82,
             "lidar_min": 0.12, "lidar_max": 3.5},
    "nav2": {},
}


def robot_model(sdf_text: str, profile: str = "real", name: str = "turtlebot3",
                pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
                lidar_noise: bool = True, report_hz: float | None = None) -> ET.Element:
    """The robot's ``<model>`` element, set to ``profile`` and placed at
    ``pose`` = (x, y, yaw), ready to embed in a world.

    ``lidar_noise=False`` zeroes Gazebo's own range noise. Gazebo cannot be
    given a seed, so its noise differs on every run; an experiment that must be
    repeatable adds the same noise itself, from a seeded generator.

    ``report_hz`` publishes the wheel odometry and the true model pose at that
    rate -- at the physics rate, every message a lockstep runner could ask for
    exists at the instant it asks -- in place of the drive's 30 Hz, whose
    stamps do not line up with the control period. It changes how often the
    state is reported, never the physics.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; one of {sorted(PROFILES)}")
    root = ET.fromstring(sdf_text)
    model = root if root.tag == "model" else root.find("model")
    if model is None:
        raise ValueError("no <model> in the robot SDF")
    model = copy.deepcopy(model)
    model.set("name", name)
    x, y, yaw = pose
    pose_el = model.find("pose")
    if pose_el is None:
        pose_el = ET.SubElement(model, "pose")
    pose_el.text = f"{x:.6f} {y:.6f} 0.01 0 0 {yaw:.6f}"

    # Meshes are referenced through ROS's package:// scheme, which plain
    # Gazebo cannot resolve; model:// against GZ_SIM_RESOURCE_PATH can.
    for uri in model.iter("uri"):
        if uri.text and uri.text.startswith("package://nav2_minimal_tb3_sim/models/"):
            uri.text = "model://" + uri.text[len("package://nav2_minimal_tb3_sim/models/"):]

    # The depth camera is not used by anything here and costs GPU time.
    for link in model.findall("link"):
        for sensor in list(link.findall("sensor")):
            if sensor.get("type") == "depth":
                link.remove(sensor)

    limits = PROFILES[profile]
    for plugin in model.findall("plugin"):
        if "diff-drive" in (plugin.get("filename") or ""):
            for kind in ("linear", "angular"):
                tag = f"max_{kind}_velocity"
                if tag in limits and plugin.find(tag) is not None:
                    plugin.find(tag).text = f"{limits[tag]}"
                    # Reverse is limited the same, as the model already had it.
                    low = plugin.find(f"min_{kind}_velocity")
                    if low is not None:
                        low.text = f"{-limits[tag]}"
    block = _lidar_block(lidar_sensor(model))
    if "lidar_min" in limits:
        block.find("range/min").text = f"{limits['lidar_min']}"
        block.find("range/max").text = f"{limits['lidar_max']}"
    if not lidar_noise and block.find("noise/stddev") is not None:
        block.find("noise/stddev").text = "0.0"
    if report_hz is not None:
        drive = next(p for p in model.findall("plugin") if "diff-drive" in (p.get("filename") or ""))
        drive.find("odom_publish_frequency").text = f"{report_hz:g}"
        pub = ET.SubElement(model, "plugin", filename="gz-sim-pose-publisher-system",
                            name="gz::sim::systems::PosePublisher")
        for tag, value in (("publish_link_pose", "false"), ("publish_model_pose", "true"),
                           ("publish_nested_model_pose", "false"),
                           ("use_pose_vector_msg", "false"), ("update_frequency", f"{report_hz:g}")):
            ET.SubElement(pub, tag).text = value
    return model


def lidar_sensor(model: ET.Element) -> ET.Element:
    """The model's lidar ``<sensor>`` element."""
    for sensor in model.iter("sensor"):
        if sensor.get("type") in ("gpu_lidar", "lidar", "gpu_ray", "ray"):
            return sensor
    raise ValueError("the robot has no lidar")


def _lidar_block(sensor: ET.Element) -> ET.Element:
    """The sensor's ``<ray>`` (SDF 1.6) or ``<lidar>`` (later SDF) element."""
    block = sensor.find("ray")
    if block is None:
        block = sensor.find("lidar")
    if block is None:
        raise ValueError("lidar sensor has neither <ray> nor <lidar>")
    return block


def lidar_spec(model: ET.Element) -> dict:
    """What the lidar is, read from the model: what a matching 2-D sensor
    needs, plus where it sits on the robot."""
    sensor = lidar_sensor(model)
    block = _lidar_block(sensor)
    scan = block.find("scan/horizontal")
    rng = block.find("range")
    noise = block.find("noise/stddev")
    link = next(lk for lk in model.findall("link") if sensor in list(lk))
    lx = float(link.find("pose").text.split()[0]) if link.find("pose") is not None else 0.0
    sx = float(sensor.find("pose").text.split()[0]) if sensor.find("pose") is not None else 0.0
    return {
        "samples": int(scan.find("samples").text),
        "min_angle": float(scan.find("min_angle").text),
        "max_angle": float(scan.find("max_angle").text),
        "range_min": float(rng.find("min").text),
        "range_max": float(rng.find("max").text),
        "noise_std": float(noise.text) if noise is not None else 0.0,
        "update_rate": float(sensor.find("update_rate").text),
        "x_offset": lx + sx,
        "topic": sensor.find("topic").text,
    }


def drive_spec(model: ET.Element) -> dict:
    """The differential drive's limits and topics, read from the model."""
    plugin = next(p for p in model.findall("plugin") if "diff-drive" in (p.get("filename") or ""))

    def val(tag: str) -> float:
        return float(plugin.find(tag).text)

    return {"max_linear_velocity": val("max_linear_velocity"),
            "max_angular_velocity": val("max_angular_velocity"),
            "max_linear_acceleration": val("max_linear_acceleration"),
            "max_angular_acceleration": val("max_angular_acceleration"),
            "wheel_separation": val("wheel_separation"),
            "wheel_radius": val("wheel_radius"),
            "cmd_vel_topic": plugin.find("topic").text,
            "odom_topic": plugin.find("odom_topic").text}
