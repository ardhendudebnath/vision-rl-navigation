"""Tests for the parts of the Nav2 SLAM comparison that do not need ROS.

The runs themselves need Nav2 and slam_toolbox inside WSL; what can be held to
account here is the arithmetic that turns them into a verdict, and the one
parameter change the SLAM arm makes to Nav2's published configuration.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import nav2_slam_comparison as cmp  # noqa: E402


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- the decision rule --------------------------------------------------------
def test_the_rule_needs_a_difference_that_is_significant_and_material():
    b = cmp.BOUND
    assert cmp.classify(0.25, [0.15, 0.35]) == "IMPLEMENTATION"
    assert cmp.classify(-0.25, [-0.35, -0.15]) == "BACKWARDS"
    assert cmp.classify(0.0, [-b / 2, b / 2]) == "PROBLEM"
    # Significant but small is not "implementation": it is bounded, and the
    # rule says so rather than reporting a trivial difference as a finding.
    assert cmp.classify(b / 2, [0.01, b - 0.01]) == "PROBLEM"
    # Too wide to say anything.
    assert cmp.classify(0.0, [-0.15, 0.15]) == "UNRESOLVED"
    # Large but not significant.
    assert cmp.classify(0.2, [-0.01, 0.4]) == "UNRESOLVED"


# --- the statistic ------------------------------------------------------------
def test_the_difference_in_costs_is_what_each_stack_loses():
    """Nav2 loses nothing, the hand-written stack loses everything, on every
    world: the difference in costs is exactly +1, and a bootstrap of a constant
    cannot move it."""
    n = 10
    ones, zeros = np.ones(n, int), np.zeros(n, int)
    with_ = {c: ones for c in cmp.CONDITIONS}
    nav_without = {c: ones for c in cmp.CONDITIONS}
    hw_without = {c: zeros for c in cmp.CONDITIONS}
    point, ci, per = cmp.pooled_did(with_, nav_without, with_, hw_without,
                                    np.random.default_rng(0), 200)
    assert point == 1.0 and ci == [1.0, 1.0]
    assert all(float(per[c].mean()) == 1.0 for c in cmp.CONDITIONS)


def test_a_controller_advantage_with_privileges_cancels_out():
    """The reason the statistic is a difference in costs. Nav2 starts ahead of
    the hand-written stack and the same worlds are taken from both, so a direct
    comparison of the two without privileges still shows Nav2's head start.
    The difference in costs removes most of it -- not all, since Nav2 has more
    successes on the dropped worlds to lose."""
    rng = np.random.default_rng(1)
    n = 50
    hw_with = {c: (rng.random(n) < 0.7).astype(int) for c in cmp.CONDITIONS}
    nav_with = {c: np.maximum(hw_with[c], (rng.random(n) < 0.7).astype(int)) for c in cmp.CONDITIONS}
    drop = {c: (rng.random(n) < 0.3) for c in cmp.CONDITIONS}
    hw_without = {c: np.where(drop[c], 0, hw_with[c]) for c in cmp.CONDITIONS}
    nav_without = {c: np.where(drop[c], 0, nav_with[c]) for c in cmp.CONDITIONS}
    point, _, _ = cmp.pooled_did(nav_with, nav_without, hw_with, hw_without,
                                 np.random.default_rng(0), 50)
    direct = float(np.mean(np.concatenate([nav_without[c] - hw_without[c] for c in cmp.CONDITIONS])))
    assert direct > 0.05, "the fixture was meant to give Nav2 a head start"
    # Worlds where Nav2 succeeded and the hand-written stack did not are the
    # only ones where the two costs can differ; the drop hits both alike.
    assert abs(point) < abs(direct)


def test_the_bootstrap_keeps_every_condition_in_every_draw():
    """Stratified: a condition with a huge effect must not vanish from a draw.
    One condition carries the whole difference; every draw must see it."""
    n = 20
    ones, zeros = np.ones(n, int), np.zeros(n, int)
    with_ = {c: ones for c in cmp.CONDITIONS}
    hw_without = {c: ones for c in cmp.CONDITIONS}
    hw_without["sparse"] = zeros
    point, ci, _ = cmp.pooled_did(with_, with_, with_, hw_without, np.random.default_rng(0), 300)
    expected = 1.0 / len(cmp.CONDITIONS)
    assert abs(point - expected) < 1e-12
    assert ci[0] == ci[1] == point, "a draw lost the condition that carries the effect"


# --- the one configuration change --------------------------------------------
def test_the_slam_arm_changes_the_global_costmap_and_nothing_else():
    make = _load(ROOT / "ros2_bridge" / "make_slam_params.py", "make_slam_params")
    published = yaml.safe_load((ROOT / "ros2_bridge" / "nav2_params.yaml").read_text(encoding="utf-8"))
    derived = make.derive(copy.deepcopy(published))
    g = derived["global_costmap"]["global_costmap"]["ros__parameters"]
    assert g["rolling_window"] is True
    assert g["width"] == g["height"] == make.WINDOW_M
    # Undo exactly those three keys and the two files must be identical.
    for key in ("rolling_window", "width", "height"):
        original = published["global_costmap"]["global_costmap"]["ros__parameters"].get(key)
        if original is None:
            del g[key]
        else:
            g[key] = original
    assert derived == published


def test_the_rolling_window_holds_the_largest_arena_from_anywhere_inside_it():
    from vision_nav.envs.splits import SHIFTS

    make = _load(ROOT / "ros2_bridge" / "make_slam_params.py", "make_slam_params")
    largest = max(max(s.get("width", 12.0), s.get("height", 12.0)) for s in SHIFTS.values())
    assert make.WINDOW_M / 2 > largest, (make.WINDOW_M, largest)
