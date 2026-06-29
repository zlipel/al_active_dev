"""
Unit tests for the diffusivity analysis script.

We cover the pure helpers — `calculate_diffusivity` (linregress over a chosen time
window) and `bash_parser` (COM-data shape / N_time adjustment) — without relying
on cluster-side md_calcs_par. The script's MSD computation itself is delegated
to md_calcs_par and is mocked by conftest.
"""
from __future__ import annotations

import numpy as np
import pytest

import analysis.process_diff_sims as pds


# ---------- calculate_diffusivity ----------

def test_calculate_diffusivity_recovers_known_slope():
    """
    For a perfectly linear MSD(t) = m*t in the [25 ns, 70 ns] window, the function
    must recover D = m/6 (3D Einstein relation).
    """
    # Times in fs: 0 -> 1e8 in 1001 steps -> covers 0..100 ns at 1e5 fs (0.1 ns) intervals.
    times = np.linspace(0.0, 1.0e8, 1001)
    m_true = 2.5e-4  # Å^2 / fs
    msd = m_true * times

    D = pds.calculate_diffusivity(msd, times)
    assert D == pytest.approx(m_true / 6.0, rel=1e-6)


def test_calculate_diffusivity_zero_slope_gives_zero():
    """Flat MSD -> D = 0."""
    times = np.linspace(0.0, 1.0e8, 1001)
    msd = np.full_like(times, 42.0)
    D = pds.calculate_diffusivity(msd, times)
    assert abs(D) < 1e-12


# ---------- bash_parser ----------

def _write_com_file(tmp_path, n_time, n_chains, base=0.0):
    """
    Write a com_mol.dat with N_time * N_chains rows of 4 columns ('chain_id x y z'),
    matching the awk filter in bash_parser. Returns the directory and the expected
    reshaped array (N_time, N_chains, 3).
    """
    rows = []
    expected = np.zeros((n_time, n_chains, 3), dtype=float)
    for t in range(n_time):
        # Header lines that the awk filter must skip (NF != 4 or non-numeric col 1).
        rows.append(f"# timestep {t}\n")
        rows.append("ITEM: chains positions\n")
        for c in range(n_chains):
            x = base + t * 0.1 + c * 0.01
            y = base + t * 0.2 + c * 0.02
            z = base + t * 0.3 + c * 0.03
            rows.append(f"{c + 1} {x} {y} {z}\n")
            expected[t, c] = [x, y, z]

    f = tmp_path / "com_mol.dat"
    f.write_text("".join(rows))
    return tmp_path, expected


def test_bash_parser_reshape_matches_synthetic_input(tmp_path):
    """bash_parser should awk-filter the 4-column lines and reshape to (N_time, N_chains, 3)."""
    n_time, n_chains = 5, 4
    out_dir, expected = _write_com_file(tmp_path, n_time, n_chains)

    arr, n_time_out = pds.bash_parser(str(out_dir), n_time, n_chains, stride=1)

    assert arr.shape == (n_time, n_chains, 3)
    assert n_time_out == n_time
    np.testing.assert_allclose(arr, expected, atol=1e-6)


def test_bash_parser_adjusts_n_time_when_input_short(tmp_path):
    """If the file has fewer rows than N_time * N_chains, the function rebinds N_time
    to floor(rows / N_chains) and reshapes accordingly. We declare 10 timesteps but
    write 5 -> expect N_time_out == 5."""
    declared_n_time, n_chains = 10, 4
    actual_n_time = 5
    out_dir, _ = _write_com_file(tmp_path, actual_n_time, n_chains)

    arr, n_time_out = pds.bash_parser(str(out_dir), declared_n_time, n_chains, stride=1)
    assert n_time_out == actual_n_time
    assert arr.shape == (actual_n_time, n_chains, 3)


def test_bash_parser_stride_thins_frames(tmp_path):
    """Stride=2 should drop every other frame from the reshape output."""
    n_time, n_chains = 6, 3
    out_dir, expected = _write_com_file(tmp_path, n_time, n_chains)

    arr, _ = pds.bash_parser(str(out_dir), n_time, n_chains, stride=2)
    np.testing.assert_allclose(arr, expected[::2], atol=1e-6)
