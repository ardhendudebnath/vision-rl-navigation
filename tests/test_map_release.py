"""Hysteresis on a cell's state: an occupied cell is freed only below a release line."""

from __future__ import annotations

import numpy as np

from vision_nav.agents.mapped import make_sensor
from vision_nav.envs.world import World, WorldConfig
from vision_nav.mapping.occupancy import FREE, OCCUPIED, OCCUPIED_AT, OccupancyMap


class _NoGeometry:
    def __init__(self):
        self.config = WorldConfig()
        self.seed = 0
        self.goal = np.array([10.0, 6.0])


class _FakeSensor:
    def __init__(self, ranges):
        self._angles = np.array([0.0])
        self._ranges = np.asarray(ranges, dtype=float)
        self.config = type("C", (), {"max_range": 6.0})()

    def scan(self, world, pose, rng=None):
        return self._ranges


POSE = np.array([1.0, 1.05, 0.0])
#: Between the 0.0 three grazes leave a once-hit cell and the -0.3 four leave it,
#: so float32 rounding of 0.9 - 3 x 0.3 cannot decide which side it falls.
RELEASE = -0.15


def _cell(m):
    return int(1.05 / m.resolution), int(3.0 / m.resolution)


def _hit_then_passes(release_at, passes):
    m = OccupancyMap(_NoGeometry(), _FakeSensor([2.0]), release_at=release_at)
    m.integrate(POSE)  # one return lands in the cell: evidence 0.9
    m.sensor = _FakeSensor([6.0])  # then beams pass straight through it
    for _ in range(passes):
        m.integrate(POSE)
    return m


def test_by_default_two_grazes_free_a_cell_one_hit_marked():
    """The published behaviour: marking and freeing share one line."""
    m = _hit_then_passes(None, 2)  # 0.9 - 0.6 = 0.3, under 0.5
    assert m.grid[_cell(m)] == FREE
    assert m.release_at == OCCUPIED_AT
    assert m.released == 1


def test_below_the_marking_line_an_occupied_cell_holds():
    m = _hit_then_passes(RELEASE, 3)  # 0.9 - 0.9 = 0.0: not yet against it
    assert m.grid[_cell(m)] == OCCUPIED
    assert m.released == 0


def test_evidence_against_it_still_frees_it():
    m = _hit_then_passes(RELEASE, 4)  # 0.9 - 1.2 = -0.3
    assert m.grid[_cell(m)] == FREE


def test_a_free_cell_still_needs_the_marking_line_to_become_occupied():
    """Hysteresis holds a state; it does not lower the bar for entering one."""
    m = OccupancyMap(_NoGeometry(), _FakeSensor([6.0]), release_at=RELEASE)
    for _ in range(2):
        m.integrate(POSE)  # seen free twice: evidence -0.6
    m.sensor = _FakeSensor([2.0])
    m.integrate(POSE)  # one hit lifts it to 0.3: inside the band, under the marking line
    assert m.evidence[_cell(m)] > RELEASE
    assert m.grid[_cell(m)] == FREE


def test_the_default_map_is_the_published_map():
    """Release at the marking line, given explicitly, changes nothing on a real
    scan sequence: grid, evidence and version agree step for step."""
    world = World(config=WorldConfig(), circles=np.array([[6.0, 6.0, 0.8], [4.0, 3.5, 0.5]]),
                  boxes=np.array([[8.0, 2.0, 8.6, 5.0]]),
                  start=np.array([2.0, 6.0, 0.0]), goal=np.array([10.0, 6.0]))
    a = OccupancyMap(world, make_sensor("lidar360", noise_std=0.05), rng=np.random.default_rng(3),
                     corroborate=True)
    b = OccupancyMap(world, make_sensor("lidar360", noise_std=0.05), rng=np.random.default_rng(3),
                     corroborate=True, release_at=OCCUPIED_AT)
    for x in np.linspace(2.0, 9.5, 25):
        pose = np.array([x, 6.0 + 0.8 * np.sin(x), 0.3 * x])
        assert a.integrate(pose) == b.integrate(pose)
        assert np.array_equal(a.grid, b.grid) and np.array_equal(a.evidence, b.evidence)
        assert a.version == b.version
