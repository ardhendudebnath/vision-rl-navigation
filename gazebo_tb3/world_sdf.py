"""A procedural world as a Gazebo Harmonic SDF world, for the virtual TurtleBot3.

The project's simulator is 2-D and analytic: circles and axis-aligned boxes in a
walled arena, a unicycle robot, a ray-cast lidar. To ask how much of what was
measured there survives a real physics engine and a real robot model -- the
nearest thing to sim-to-real without hardware -- each world is rebuilt in
Gazebo exactly: every circle a static cylinder, every box a static box, the
arena boundary four walls whose inner faces sit on its edges, all tall enough
that a lidar mounted on any TurtleBot3 sees them as the 2-D sensor does.

Kept free of ROS and Gazebo imports so the geometry can be tested anywhere.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

import numpy as np

#: Obstacle height, metres. Above every TurtleBot3 lidar mount (~0.17-0.19 m),
#: so a beam that hits an obstacle in 2-D hits it here too.
OBSTACLE_HEIGHT = 0.5
#: Thickness of the arena walls, which sit outside the arena so their inner
#: faces are its boundary.
WALL_THICKNESS = 0.1
#: Physics step, seconds. The control period (0.1 s) is a whole number of these.
STEP_SIZE = 0.002


def _box(name: str, cx: float, cy: float, sx: float, sy: float, colour: str) -> str:
    size = f"{sx:.6f} {sy:.6f} {OBSTACLE_HEIGHT:.6f}"
    return f"""
    <model name="{escape(name)}">
      <static>true</static>
      <pose>{cx:.6f} {cy:.6f} {OBSTACLE_HEIGHT / 2:.6f} 0 0 0</pose>
      <link name="link">
        <collision name="collision"><geometry><box><size>{size}</size></box></geometry></collision>
        <visual name="visual"><geometry><box><size>{size}</size></box></geometry>
          <material><ambient>{colour}</ambient><diffuse>{colour}</diffuse></material></visual>
      </link>
    </model>"""


def _cylinder(name: str, cx: float, cy: float, r: float) -> str:
    geom = f"<cylinder><radius>{r:.6f}</radius><length>{OBSTACLE_HEIGHT:.6f}</length></cylinder>"
    colour = "0.55 0.35 0.2 1"
    return f"""
    <model name="{escape(name)}">
      <static>true</static>
      <pose>{cx:.6f} {cy:.6f} {OBSTACLE_HEIGHT / 2:.6f} 0 0 0</pose>
      <link name="link">
        <collision name="collision"><geometry>{geom}</geometry></collision>
        <visual name="visual"><geometry>{geom}</geometry>
          <material><ambient>{colour}</ambient><diffuse>{colour}</diffuse></material></visual>
      </link>
    </model>"""


def obstacles(world) -> list[dict]:
    """Every static solid in the world, as plain shapes in world coordinates.

    Returned separately from the SDF text so tests can check the geometry
    without parsing XML, and so the walls' placement is stated once.
    """
    w, h = float(world.config.width), float(world.config.height)
    t = WALL_THICKNESS
    shapes = [
        {"name": "wall_south", "kind": "box", "x": w / 2, "y": -t / 2, "sx": w + 2 * t, "sy": t},
        {"name": "wall_north", "kind": "box", "x": w / 2, "y": h + t / 2, "sx": w + 2 * t, "sy": t},
        {"name": "wall_west", "kind": "box", "x": -t / 2, "y": h / 2, "sx": t, "sy": h + 2 * t},
        {"name": "wall_east", "kind": "box", "x": w + t / 2, "y": h / 2, "sx": t, "sy": h + 2 * t},
    ]
    for i, (x, y, r) in enumerate(np.asarray(world.circles, dtype=float).reshape(-1, 3)):
        shapes.append({"name": f"circle_{i}", "kind": "cylinder", "x": x, "y": y, "r": r})
    for i, (x0, y0, x1, y1) in enumerate(np.asarray(world.boxes, dtype=float).reshape(-1, 4)):
        shapes.append({"name": f"box_{i}", "kind": "box", "x": (x0 + x1) / 2, "y": (y0 + y1) / 2,
                       "sx": x1 - x0, "sy": y1 - y0})
    return shapes


def world_sdf(world, robot_xml: str | None = None, name: str = "vision_nav",
              sensors: bool = True) -> str:
    """The SDF text of ``world``, with the robot embedded.

    ``robot_xml`` is the robot's ``<model>`` element as text, already placed
    at the start pose (:func:`robot_sdf.robot_model` does both); ``None``
    leaves the robot out. The world carries the systems a lockstep runner
    needs: physics, sensors (rendered with ogre2, which the GPU lidar
    requires), scene broadcasting -- whose pose stream is how the true pose is
    read for scoring, never for control -- and user commands. ``sensors=False``
    leaves the rendering system out: no lidar, physics and odometry only.
    """
    sensor_system = """
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>""" if sensors else ""
    parts = []
    for s in obstacles(world):
        if s["kind"] == "cylinder":
            parts.append(_cylinder(s["name"], s["x"], s["y"], s["r"]))
        else:
            colour = "0.4 0.4 0.45 1" if s["name"].startswith("wall") else "0.3 0.45 0.6 1"
            parts.append(_box(s["name"], s["x"], s["y"], s["sx"], s["sy"], colour))
    robot = f"\n    {robot_xml}" if robot_xml is not None else ""
    gx, gy = (float(v) for v in world.goal)
    w, h = float(world.config.width), float(world.config.height)
    return f"""<?xml version="1.0"?>
<sdf version="1.9">
  <world name="{escape(name)}">
    <physics name="lockstep" type="ignored">
      <max_step_size>{STEP_SIZE}</max_step_size>
      <real_time_factor>0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>{sensor_system}
    <scene><ambient>0.6 0.6 0.6 1</ambient><background>0.8 0.85 0.9 1</background></scene>
    <light type="directional" name="sun">
      <cast_shadows>false</cast_shadows>
      <pose>{w / 2:.3f} {h / 2:.3f} 10 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <direction>-0.3 0.2 -1</direction>
    </light>
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>{w + 4:.1f} {h + 4:.1f}</size></plane></geometry>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>{w + 4:.1f} {h + 4:.1f}</size></plane></geometry>
          <material><ambient>0.85 0.85 0.8 1</ambient><diffuse>0.85 0.85 0.8 1</diffuse></material>
        </visual>
      </link>
    </model>
    <model name="goal_marker">
      <static>true</static>
      <pose>{gx:.6f} {gy:.6f} 0.002 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry><cylinder><radius>{float(world.config.goal_tolerance):.3f}</radius><length>0.002</length></cylinder></geometry>
          <material><ambient>0.1 0.8 0.2 1</ambient><diffuse>0.1 0.8 0.2 1</diffuse></material>
        </visual>
      </link>
    </model>{''.join(parts)}{robot}
  </world>
</sdf>
"""
