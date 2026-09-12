"""Tests for the perception audit.

This measurement is meant to replace an unreliable end-to-end inference, so
it has to be right. The failure mode that would matter most is silently
overstating detection -- that would "confirm" the perception hypothesis for
the wrong reason.
"""

from __future__ import annotations

import numpy as np
import pytest

from vision_nav.analysis.perception import audit_pose, find_gaps, traversable_mask
from vision_nav.envs.world import World, WorldConfig, generate_world


def empty_world(size: float = 20.0) -> World:
    return World(
        config=WorldConfig(width=size, height=size),
        circles=np.zeros((0, 3)),
        boxes=np.zeros((0, 4)),
        start=np.zeros(3),
        goal=np.zeros(2),
    )


# ----------------------------------------------------------------------
# find_gaps
# ----------------------------------------------------------------------
def test_no_traversable_directions_gives_no_gaps():
    assert find_gaps(np.zeros(16, dtype=bool)) == []


def test_all_traversable_is_one_gap():
    assert find_gaps(np.ones(16, dtype=bool)) == [(0, 15)]


def test_single_contiguous_run():
    mask = np.zeros(16, dtype=bool)
    mask[4:9] = True
    assert find_gaps(mask) == [(4, 8)]


def test_two_separate_runs():
    mask = np.zeros(16, dtype=bool)
    mask[2:4] = True
    mask[10:13] = True
    assert sorted(find_gaps(mask)) == [(2, 3), (10, 12)]


def test_wrapping_run_is_one_gap_not_two():
    """A scan is a circle; a gap across the 0/2pi seam must not be split."""
    mask = np.zeros(16, dtype=bool)
    mask[14:] = True
    mask[:2] = True
    gaps = find_gaps(mask)
    assert len(gaps) == 1
    assert gaps[0] == (14, 1)


# ----------------------------------------------------------------------
# traversable_mask
# ----------------------------------------------------------------------
def test_open_space_is_traversable_in_every_direction():
    world = empty_world()
    angles = np.linspace(-np.pi, np.pi, 64, endpoint=False)
    mask = traversable_mask(world, np.array([10.0, 10.0]), angles, probe_distance=1.0)
    assert mask.all()


def test_a_wall_blocks_the_directions_behind_it():
    world = World(
        config=WorldConfig(width=20.0, height=20.0),
        circles=np.zeros((0, 3)),
        boxes=np.array([[11.0, 5.0, 12.0, 15.0]]),
        start=np.zeros(3),
        goal=np.zeros(2),
    )
    angles = np.array([0.0, np.pi])  # +x is into the wall, -x is away
    mask = traversable_mask(world, np.array([10.0, 10.0]), angles, probe_distance=2.0)
    assert not mask[0]
    assert mask[1]


def test_probe_distance_matters():
    """A direction can be traversable for 0.5 m but not for 3 m."""
    world = World(
        config=WorldConfig(width=20.0, height=20.0),
        circles=np.array([[13.0, 10.0, 0.5]]),
        boxes=np.zeros((0, 4)),
        start=np.zeros(3),
        goal=np.zeros(2),
    )
    angles = np.array([0.0])
    pos = np.array([10.0, 10.0])
    assert traversable_mask(world, pos, angles, probe_distance=1.0)[0]
    assert not traversable_mask(world, pos, angles, probe_distance=3.0)[0]


# ----------------------------------------------------------------------
# audit_pose
# ----------------------------------------------------------------------
def test_open_space_is_fully_detected_at_any_resolution():
    world = empty_world()
    pose = np.array([10.0, 10.0, 0.0])
    for n in (32, 64, 128):
        stats = audit_pose(world, pose, n_beams=n, probe_distance=1.0)
        assert stats.n_gaps == 1
        assert stats.detection_rate == 1.0


def test_a_coarse_reference_is_refused():
    """The ground truth must out-resolve the sensor, or detection is inflated."""
    world = empty_world()
    pose = np.array([10.0, 10.0, 0.0])
    with pytest.raises(ValueError, match="too coarse"):
        audit_pose(world, pose, n_beams=128, ground_truth_rays=256)


def test_a_robot_width_slot_is_missed_at_low_resolution():
    """The mechanism under test: a real opening with no beam pointed into it.

    Two circles leave a 0.60 m slot -- comfortably wider than the 0.44 m robot
    -- centred 4 m ahead. At that range the slot subtends only ~8.6 degrees,
    below the 11.25 degree spacing of a 32-beam scan, so whether it is seen
    depends purely on where the beams happen to fall. Sweeping the robot's
    heading samples that phase.
    """
    cfg = WorldConfig(width=20.0, height=20.0, robot_radius=0.22)
    # Circles of radius 1.0 centred +/-1.3 m off-axis leave 0.6 m between them.
    world = World(
        config=cfg,
        circles=np.array([[9.0, 11.3, 1.0], [9.0, 8.7, 1.0]]),
        boxes=np.zeros((0, 4)),
        start=np.zeros(3),
        goal=np.zeros(2),
    )
    slot_width = 2.6 - 2 * 1.0
    assert slot_width > 2 * cfg.robot_radius, "slot must be traversable at all"

    # The probe must be long enough to actually reach past the obstacles.
    # Too short and every direction is traversable, so there is no slot to
    # miss and the audit degenerates to one all-encompassing gap.
    probe = 3.0
    headings = np.linspace(-np.pi, np.pi, 60, endpoint=False)
    hits = {}
    for n in (32, 256):
        hits[n] = sum(
            audit_pose(
                world, np.array([7.0, 10.0, h]), n_beams=n, probe_distance=probe
            ).n_detected
            for h in headings
        )

    # A fine scan finds strictly more gap-instances than a coarse one: the
    # perception limit made visible without training anything.
    assert hits[256] > hits[32], f"expected finer scan to see more: {hits}"


def test_detection_is_monotone_in_beam_count_on_real_worlds():
    """More beams can only reveal more gaps, never fewer.

    Not a tautology -- it is the property that makes the audit meaningful, and
    a bug in the beam-to-gap indexing would break it.
    """
    for seed in range(4):
        world = generate_world(seed)
        pose = world.start
        rates = [
            audit_pose(world, pose, n_beams=n, probe_distance=1.0).n_detected
            for n in (16, 32, 64, 128)
        ]
        assert rates == sorted(rates), f"non-monotone detection on seed {seed}: {rates}"


def test_stats_are_self_consistent():
    world = generate_world(1)
    stats = audit_pose(world, world.start, n_beams=32, probe_distance=1.0)
    assert stats.n_gaps == len(stats.widths) == len(stats.detected)
    assert stats.n_detected == sum(stats.detected)
    assert 0.0 <= stats.detection_rate <= 1.0
    assert sum(stats.widths) <= 2 * np.pi + 1e-9
