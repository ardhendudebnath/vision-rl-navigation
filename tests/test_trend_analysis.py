"""Tests for the monotone-trend test used on the FOV sweep.

The statistics here decide a headline claim, so the machinery needs to be
right. Two things would be quietly damaging: tie handling (every level is
repeated six times, so untied ranks would silently distort rho), and a
Monte Carlo p that can reach exactly zero and overstate the evidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from trend_analysis import permutation_trend_p, rankdata, spearman  # noqa: E402


# ----------------------------------------------------------------------
# rankdata
# ----------------------------------------------------------------------
def test_ranks_of_distinct_values():
    assert rankdata(np.array([10.0, 30.0, 20.0])).tolist() == [1.0, 3.0, 2.0]


def test_ties_receive_the_average_rank():
    """Levels repeat six times each; untied ranks would distort rho."""
    assert rankdata(np.array([5.0, 5.0, 9.0])).tolist() == [1.5, 1.5, 3.0]
    assert rankdata(np.array([1.0, 1.0, 1.0, 1.0])).tolist() == [2.5] * 4


def test_rank_sum_is_preserved_under_tie_averaging():
    a = np.array([2.0, 2.0, 7.0, 7.0, 7.0, 9.0])
    assert rankdata(a).sum() == pytest.approx(len(a) * (len(a) + 1) / 2)


# ----------------------------------------------------------------------
# spearman
# ----------------------------------------------------------------------
def test_perfect_monotone_relationships():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman(x, np.array([1.0, 2.0, 3.0, 4.0])) == pytest.approx(1.0)
    assert spearman(x, np.array([9.0, 7.0, 5.0, 3.0])) == pytest.approx(-1.0)


def test_spearman_is_rank_based_not_value_based():
    """A monotone but wildly nonlinear response is still rho = 1."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman(x, np.array([1.0, 2.0, 4.0, 10_000.0])) == pytest.approx(1.0)


def test_constant_input_gives_zero_not_nan():
    x = np.array([1.0, 1.0, 1.0, 1.0])
    assert spearman(x, np.array([3.0, 1.0, 4.0, 2.0])) == 0.0


def test_handles_the_sweep_shape():
    """Four levels x six seeds, the actual FOV sweep layout."""
    levels = np.repeat([90.0, 180.0, 270.0, 360.0], 6)
    rng = np.random.default_rng(0)
    rising = np.repeat([0.58, 0.61, 0.64, 0.68], 6) + rng.normal(0, 0.01, 24)
    assert spearman(levels, rising) > 0.8


# ----------------------------------------------------------------------
# permutation p
# ----------------------------------------------------------------------
def test_strong_trend_is_significant():
    levels = np.repeat([90.0, 180.0, 270.0, 360.0], 6)
    rng = np.random.default_rng(1)
    values = np.repeat([0.55, 0.62, 0.68, 0.75], 6) + rng.normal(0, 0.02, 24)
    rho, p = permutation_trend_p(levels, values, n_perm=2000, seed=0)
    assert rho > 0.8
    assert p < 0.01


def test_pure_noise_is_not_significant():
    levels = np.repeat([90.0, 180.0, 270.0, 360.0], 6)
    values = np.random.default_rng(7).normal(0.6, 0.05, 24)
    _, p = permutation_trend_p(levels, values, n_perm=2000, seed=0)
    assert p > 0.05


def test_p_can_never_be_exactly_zero():
    """Add-one correction: a Monte Carlo p of 0 would overstate the evidence."""
    levels = np.repeat([1.0, 2.0, 3.0, 4.0], 6)
    values = np.arange(24, dtype=float)  # perfectly monotone
    _, p = permutation_trend_p(levels, values, n_perm=500, seed=0)
    assert p > 0.0
    assert p == pytest.approx(1 / 501, abs=1e-9)


def test_p_is_reproducible_for_a_fixed_rng_seed():
    levels = np.repeat([1.0, 2.0, 3.0, 4.0], 6)
    values = np.random.default_rng(3).normal(0, 1, 24)
    a = permutation_trend_p(levels, values, n_perm=1000, seed=42)
    b = permutation_trend_p(levels, values, n_perm=1000, seed=42)
    assert a == b


def test_p_is_two_sided():
    """A strong negative trend must be as significant as a positive one."""
    levels = np.repeat([1.0, 2.0, 3.0, 4.0], 6)
    rng = np.random.default_rng(5)
    up = np.repeat([0.5, 0.6, 0.7, 0.8], 6) + rng.normal(0, 0.02, 24)
    _, p_up = permutation_trend_p(levels, up, n_perm=2000, seed=0)
    _, p_down = permutation_trend_p(levels, -up, n_perm=2000, seed=0)
    assert p_up == pytest.approx(p_down, abs=0.02)
