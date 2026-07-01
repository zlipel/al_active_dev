"""
Pure-logic unit tests for the EOS analysis helpers.

The bootstrap / EOS root-finding code in process_eos_sims is testable on its own
because the math (block-average error, root finding, integral inversion) is
independent of LAMMPS file format. The end-to-end paths that read thermo.avg
are exercised indirectly by the fixtures.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

import analysis.process_eos_sims as eos


# ---------- split_error ----------

def test_split_error_constant_array_zero_error():
    """Constant array has zero variance across blocks -> standard error = 0."""
    a = np.full(20, 3.14)
    std, means = eos.split_error(a, n=5)
    assert std == pytest.approx(0.0, abs=1e-12)
    assert len(means) == 5
    np.testing.assert_allclose(means, 3.14)


def test_split_error_block_means_match_manual_calc():
    """Block means must match a manual split with the same divmod logic."""
    a = np.arange(12, dtype=float)
    std, means = eos.split_error(a, n=4)
    # 12/4 = 3 rows per block, no remainder
    expected = [np.mean(a[i*3:(i+1)*3]) for i in range(4)]
    np.testing.assert_allclose(means, expected)
    # SE of the means
    assert std == pytest.approx(np.std(expected) / np.sqrt(4))


def test_split_error_handles_uneven_split():
    """When len(a) is not divisible by n, divmod distributes the remainder
    over the early blocks (one extra element each)."""
    a = np.arange(10, dtype=float)  # 10 / 3 -> 3 blocks of sizes [4, 3, 3]
    std, means = eos.split_error(a, n=3)
    assert len(means) == 3
    # First block has 4 elements, others have 3
    assert means[0] == pytest.approx(np.mean(a[0:4]))
    assert means[1] == pytest.approx(np.mean(a[4:7]))
    assert means[2] == pytest.approx(np.mean(a[7:10]))


# ---------- calc_exp_density ----------

def test_calc_exp_density_returns_minus_one_when_work_not_reached():
    """If the integral never exceeds `work`, the function signals failure with -1."""
    # Build a spline whose integral of cs(z)/z^2 stays small.
    rho = np.array([0.0, 0.1, 0.5, 1.0])
    P = np.array([0.0, 1e-4, 1e-4, 1e-4])
    cs = CubicSpline(rho, P)
    out = eos.calc_exp_density(cs, rhomin=0.1, rhomax=1.0, work=10.0)
    assert out == -1


def test_calc_exp_density_returns_root_when_work_reached():
    """If the integral exceeds work for some rho, return that rho > 0."""
    rho = np.array([0.0, 0.1, 0.5, 1.0])
    # Big positive pressures so the running integral exceeds work quickly.
    P = np.array([0.0, 100.0, 200.0, 500.0])
    cs = CubicSpline(rho, P)
    out = eos.calc_exp_density(cs, rhomin=0.1, rhomax=1.0, work=0.1)
    assert out > 0
    assert out <= 1.0 + 0.2  # function searches up to rhomax + 0.2


# ---------- find_highest_root ----------

def test_find_highest_root_returns_zero_when_no_crossing():
    """All-positive pressures -> no root -> 0."""
    rho = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    P = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    err = np.zeros_like(P)
    assert eos.find_highest_root(P, rho, err) == 0


def test_find_highest_root_picks_max_root_when_multiple_crossings():
    """If pressure crosses zero multiple times, the highest-density root wins."""
    rho = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    # Sign pattern: +,-,+,-,+,+ -> roots between (0.1,0.2), (0.2,0.3), (0.3,0.4), (0.4,0.5)
    P = np.array([1.0, -1.0, 1.0, -1.0, 1.0, 1.0])
    err = np.zeros_like(P)
    out = eos.find_highest_root(P, rho, err)
    assert 0.4 <= out <= 0.5, f"expected root in (0.4, 0.5), got {out}"


