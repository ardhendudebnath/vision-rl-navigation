"""Direct measurement of what a discrete lidar can and cannot see.

Phase 2f inferred a perception limit from end-to-end success rate, and Phase
2g showed that inference was swamped by training-seed noise. This module
measures the mechanism itself instead: given a world and a pose, how often
does an N-beam scan fail to reveal an opening the robot could actually drive
through?

That question involves no policy, no training and no seeds, so it answers
"can the sensor see the gap?" cleanly and separately from "does the policy
use what it sees?" -- the two hypotheses the end-to-end experiment could not
distinguish.

Definitions used throughout:

*Traversable direction*
    A heading along which the robot's disc can translate ``probe_distance``
    metres without collision. This is checked against the world geometry
    directly, not against any sensor.

*Gap*
    A maximal circular run of traversable directions.

*Detected gap*
    A gap containing at least one lidar beam whose measured range is long
    enough to indicate free space. A gap with no beam inside it, or whose
    beams all terminate on the flanking obstacles, is invisible to the policy
    however good the policy is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vision_nav.envs.sensors import Lidar2D, LidarConfig

__all__ = ["GapStats", "traversable_mask", "find_gaps", "audit_pose"]


@dataclass
class GapStats:
    """Detection outcome for one pose."""

    n_gaps: int
    n_detected: int
    #: Angular width of each gap, radians.
    widths: list[float]
    #: Whether each gap was detected, aligned with ``widths``.
    detected: list[bool]

    @property
    def detection_rate(self) -> float:
        return self.n_detected / self.n_gaps if self.n_gaps else float("nan")


def traversable_mask(
    world,
    position: np.ndarray,
    angles: np.ndarray,
    probe_distance: float,
    n_samples: int = 12,
) -> np.ndarray:
    """Which of ``angles`` the robot can translate ``probe_distance`` along.

    Checked against world geometry, so this is ground truth: it is what the
    robot could do, not what any sensor reports.
    """
    position = np.asarray(position, dtype=np.float64)
    radius = world.config.robot_radius

    # (n_angles, n_samples, 2) points along each candidate heading.
    steps = np.linspace(probe_distance / n_samples, probe_distance, n_samples)
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    pts = position[None, None, :] + dirs[:, None, :] * steps[None, :, None]

    return np.all(world.clearance(pts) > radius, axis=1)


def find_gaps(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal circular runs of ``True``, as inclusive ``(start, end)`` indices.

    Wraps around the end of the array, because a scan is a circle: a gap
    straddling the 0/2pi boundary is one gap, not two.
    """
    n = len(mask)
    if not mask.any():
        return []
    if mask.all():
        return [(0, n - 1)]

    # Rotate so index 0 starts a run, making the wrap case ordinary.
    start_offset = int(np.argmax(~mask & np.roll(mask, -1))) + 1
    rolled = np.roll(mask, -start_offset)

    gaps = []
    i = 0
    while i < n:
        if rolled[i]:
            j = i
            while j + 1 < n and rolled[j + 1]:
                j += 1
            gaps.append(((i + start_offset) % n, (j + start_offset) % n))
            i = j + 1
        else:
            i += 1
    return gaps


def audit_pose(
    world,
    pose: np.ndarray,
    n_beams: int,
    probe_distance: float = 1.0,
    ground_truth_rays: int = 2048,
    max_range: float = 6.0,
) -> GapStats:
    """Measure gap detection for one sensor resolution at one pose.

    Parameters
    ----------
    n_beams:
        Beam count of the sensor under test.
    probe_distance:
        How far the robot must be able to travel for a heading to count as
        traversable.
    ground_truth_rays:
        Angular resolution of the ground-truth sweep. Must be much finer than
        ``n_beams`` or the "truth" inherits the sensor's own blind spots.
    """
    if ground_truth_rays < 8 * n_beams:
        raise ValueError(
            f"ground_truth_rays={ground_truth_rays} is too coarse to audit "
            f"n_beams={n_beams}; the reference would share the sensor's blind "
            "spots and detection would be overstated"
        )

    position = np.asarray(pose[:2], dtype=np.float64)
    dense_angles = np.linspace(-np.pi, np.pi, ground_truth_rays, endpoint=False)
    mask = traversable_mask(world, position, dense_angles, probe_distance)

    lidar = Lidar2D(LidarConfig(n_beams=n_beams, max_range=max_range))
    beam_angles = lidar.config.beam_angles() + float(pose[2])
    ranges = lidar.scan(world, pose)

    # A beam "shows free space" if it reaches past what the robot needs to
    # traverse: the probe distance plus its own radius.
    clear_enough = ranges >= probe_distance + world.config.robot_radius
    # Index each beam into the dense angular grid.
    wrapped = (beam_angles + np.pi) % (2.0 * np.pi) - np.pi
    beam_idx = np.round((wrapped + np.pi) / (2 * np.pi) * ground_truth_rays).astype(int)
    beam_idx %= ground_truth_rays

    widths: list[float] = []
    detected: list[bool] = []
    step = 2.0 * np.pi / ground_truth_rays

    for start, end in find_gaps(mask):
        span = (end - start) % ground_truth_rays + 1
        widths.append(span * step)
        if start <= end:
            inside = (beam_idx >= start) & (beam_idx <= end)
        else:  # wrapped gap
            inside = (beam_idx >= start) | (beam_idx <= end)
        detected.append(bool(np.any(inside & clear_enough)))

    return GapStats(
        n_gaps=len(widths),
        n_detected=int(sum(detected)),
        widths=widths,
        detected=detected,
    )
