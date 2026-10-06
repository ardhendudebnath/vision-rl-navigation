"""A correlative front end, built the way slam_toolbox builds its own.

Every published result here localises with :class:`ScanMatcher`
(:mod:`vision_nav.mapping.localisation`): each scan, every control period, is
matched against the whole occupancy map inside a +/-0.08 m, +/-0.024 rad window,
with a motion prior strong enough that about 0.03 m of match noise cannot be
taken seriously. §9.17 and §9.18 found what that costs under range noise. The
drift outgrows the window -- the failing `noisy_lidar` episodes sit 0.318 m from
the truth, four times its reach -- and from then on the matcher can only agree
with a map the same drift built, while the back end cannot see the drift either,
because every closure it can make links two keyframes that drifted together.

slam_toolbox reaches 1.000 on `noisy_lidar`, and it was run here with the
parameters in ``ros2_bridge/slam_params.yaml``. Its front end differs from this
stack's in three ways, and this copies each, with those values:

**It matches on keyframes, not every scan.** A match is made only when the robot
has travelled ``minimum_travel_distance`` (0.5 m) or turned
``minimum_travel_heading`` (0.5 rad) since the last one; in between, the
odometry carries the pose. Matching every scan at 10 Hz gives match noise ten
chances a metre to accumulate, which is what the published matcher's prior
exists to suppress.

**It matches against recent scans, not the map.** The target is a buffer of the
last ``scan_buffer_size`` (10) keyframe scans within
``scan_buffer_maximum_scan_distance`` (10 m), each placed at its own corrected
pose and smeared by ``correlation_search_space_smear_deviation`` (0.1 m). A
buffer drifts too, but only by what accumulated over its last few metres, where
the whole map holds every metre of drift since the start.

**Its search is wide.** ``correlation_search_space_dimension`` is 0.5 m, so
+/-0.25 m, and headings are searched over ``coarse_search_angle_offset``
(+/-0.349 rad) in steps of ``coarse_angle_resolution`` (0.0349 rad) before a
fine pass at ``fine_search_angle_offset`` (0.00349 rad). The heading search
alone is fifteen times this stack's. Karto's tie-break towards the odometry is
deliberately weak -- a multiplicative penalty of ``1 - 0.2 d^2 / variance``,
floored at ``minimum_distance_penalty`` (0.5) and ``minimum_angle_penalty``
(0.9) -- and copied as it is.

The spatial search is coarse-to-fine here where slam_toolbox searches the whole
window at 0.01 m, because this is NumPy and that is C++; the fine pass covers
the coarse step at the same 0.01 m, so the reachable poses are the same.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vision_nav.envs.robot import wrap_angle
from vision_nav.mapping.posegraph import (
    _grid,
    _offsets,
    _project,
    _sample,
    blurred_field,
    relative_pose,
    scan_points,
)

__all__ = ["CorrelativeConfig", "CorrelativeFrontEnd"]


@dataclass
class CorrelativeConfig:
    """slam_toolbox's scan-matcher parameters, as this repository ran them."""

    #: ``minimum_travel_distance`` and ``minimum_travel_heading``.
    keyframe_distance: float = 0.5
    keyframe_angle: float = 0.5
    #: ``scan_buffer_size`` and ``scan_buffer_maximum_scan_distance``.
    buffer_size: int = 10
    buffer_distance: float = 10.0
    #: Half of ``correlation_search_space_dimension``, searched coarse then at
    #: ``correlation_search_space_resolution``.
    window: float = 0.25
    coarse_step: float = 0.05
    fine_step: float = 0.01
    #: ``coarse_search_angle_offset``, ``coarse_angle_resolution`` and
    #: ``fine_search_angle_offset``.
    angular_window: float = 0.349
    coarse_angular_step: float = 0.0349
    fine_angular_step: float = 0.00349
    #: ``correlation_search_space_smear_deviation``, and the lookup grid's cell.
    smear: float = 0.1
    resolution: float = 0.02
    #: Returns used per match. The search scores thousands of candidate poses,
    #: and the fit is decided by the shape of the scan, not every last beam.
    max_points: int = 180
    #: ``distance_variance_penalty``, ``angle_variance_penalty``,
    #: ``minimum_distance_penalty`` and ``minimum_angle_penalty``; the 0.2 gain
    #: is Karto's own constant.
    distance_variance_penalty: float = 0.5
    angle_variance_penalty: float = 1.0
    minimum_distance_penalty: float = 0.5
    minimum_angle_penalty: float = 0.9
    penalty_gain: float = 0.2
    #: ``link_match_minimum_response_fine``: below this the match is refused.
    minimum_response: float = 0.1
    #: Too few points on either side to say anything.
    min_points: int = 20


class CorrelativeFrontEnd:
    """Keyframe scans matched against a buffer of recent keyframe scans."""

    def __init__(self, config: CorrelativeConfig | None = None) -> None:
        self.config = config or CorrelativeConfig()
        self.reset()

    def reset(self) -> None:
        #: ``(pose, scan in the robot's own frame)`` for recent keyframes.
        self.buffer: list[tuple[np.ndarray, np.ndarray]] = []
        #: Diagnostics for the episode.
        self.matches = 0
        self.refused = 0
        self.total_shift = 0.0
        self.last_shift = 0.0
        self.last_response = 0.0

    # ------------------------------------------------------------------
    def due(self, pose: np.ndarray) -> bool:
        """Whether ``pose`` is far enough from the last keyframe to match."""
        if not self.buffer:
            return True
        rel = relative_pose(self.buffer[-1][0], np.asarray(pose, dtype=float))
        return (float(np.hypot(rel[0], rel[1])) >= self.config.keyframe_distance
                or abs(float(rel[2])) >= self.config.keyframe_angle)

    def update(self, estimate: np.ndarray, ranges: np.ndarray, sensor) -> np.ndarray:
        """The pose after this scan: matched if it is a keyframe, else as given."""
        estimate = np.asarray(estimate, dtype=float).copy()
        self.last_shift = 0.0
        if not self.due(estimate):
            return estimate
        local = scan_points(ranges, sensor)
        if self.buffer and len(local) >= self.config.min_points:
            matched = self._match(estimate, local)
            if matched is not None:
                self.last_shift = float(np.linalg.norm(matched[:2] - estimate[:2]))
                self.total_shift += self.last_shift
                estimate = matched
        self._remember(estimate, local)
        return estimate

    # ------------------------------------------------------------------
    def _remember(self, pose: np.ndarray, local: np.ndarray) -> None:
        self.buffer.append((pose.copy(), local))
        if len(self.buffer) > self.config.buffer_size:
            self.buffer.pop(0)

    def _target(self, pose: np.ndarray) -> np.ndarray:
        """The buffered scans near ``pose``, placed in the world at their poses."""
        cfg = self.config
        parts = [_project(p, s) for p, s in self.buffer
                 if len(s) and float(np.linalg.norm(p[:2] - pose[:2])) <= cfg.buffer_distance]
        return np.concatenate(parts) if parts else np.zeros((0, 2))

    def _match(self, prior: np.ndarray, local: np.ndarray) -> np.ndarray | None:
        cfg = self.config
        target = self._target(prior)
        if len(target) < cfg.min_points:
            self.refused += 1
            return None
        source = local
        if len(source) > cfg.max_points:
            source = source[np.linspace(0, len(source) - 1, cfg.max_points).astype(int)]
        pad = cfg.window + 3.0 * cfg.smear + cfg.resolution
        field, origin = blurred_field(target, cfg.smear, cfg.resolution, pad)
        best, _ = self._search(prior, prior, source, field, origin, cfg.window,
                               cfg.coarse_step, cfg.angular_window, cfg.coarse_angular_step)
        best, response = self._search(prior, best, source, field, origin, cfg.coarse_step,
                                      cfg.fine_step, cfg.coarse_angular_step,
                                      cfg.fine_angular_step)
        self.last_response = response
        if response < cfg.minimum_response:
            self.refused += 1
            return None
        self.matches += 1
        return best

    def _search(self, prior, centre, source, field, origin, window, step,
                angular_window, angular_step) -> tuple[np.ndarray, float]:
        """Best pose by penalised response, and its unpenalised response."""
        cfg = self.config
        offsets = _grid(window, step)
        dthetas = _offsets(angular_window, angular_step)
        th = centre[2] + dthetas
        cos, sin = np.cos(th)[:, None], np.sin(th)[:, None]
        x = cos * source[None, :, 0] - sin * source[None, :, 1]
        y = sin * source[None, :, 0] + cos * source[None, :, 1]
        pts = np.stack([x, y], axis=-1) + centre[:2]
        cand = pts[:, None, :, :] + offsets[None, :, None, :]
        response = _sample(field, cand, cfg.resolution, origin).mean(axis=-1)   # (T, K)
        # Karto's tie-break towards the odometry: weak, multiplicative, floored.
        dxy = centre[:2] + offsets - prior[:2]
        dist = np.maximum(1.0 - cfg.penalty_gain * np.sum(dxy * dxy, axis=-1)
                          / cfg.distance_variance_penalty, cfg.minimum_distance_penalty)
        da = wrap_angle(centre[2] + dthetas - prior[2])
        ang = np.maximum(1.0 - cfg.penalty_gain * da * da / cfg.angle_variance_penalty,
                         cfg.minimum_angle_penalty)
        score = response * dist[None, :] * ang[:, None]
        t, k = np.unravel_index(int(np.argmax(score)), score.shape)
        out = centre.copy()
        out[:2] = centre[:2] + offsets[k]
        out[2] = wrap_angle(centre[2] + dthetas[t])
        return out, float(response[t, k])
