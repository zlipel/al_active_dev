"""
Unit tests for the pure helpers in `simulation/make_diff.py`.

Targets:
  - `SimulationManager.get_mpipi_charge` — single-residue check over DEHKR
  - `SimulationManager.check_complete`   — filesystem-based completion gate
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from simulation.make_diff import SimulationManager


# ---------- get_mpipi_charge ----------

@pytest.mark.parametrize(
    "seq, expected",
    [
        ("",                 0),  # empty
        ("AGSVLI",           0),  # only neutral residues
        ("AGGD",             1),  # single D
        ("KKKRRR",           1),  # all charged
        ("AGGE",             1),  # E
        ("AGGH",             1),  # H
        ("AGGK",             1),  # K
        ("AGGR",             1),  # R
        ("agg d",            0),  # lowercase d not matched, space skipped
    ],
)
def test_get_mpipi_charge(seq, expected):
    assert SimulationManager.get_mpipi_charge(seq) == expected


# ---------- check_complete ----------

def _bare_manager():
    """Build a SimulationManager without running __init__ (which expects a config)."""
    return SimulationManager.__new__(SimulationManager)


def _write_com_mol(path: Path, n_rows: int) -> None:
    """Write a minimal com_mol.dat with `n_rows` numeric rows (col 0 = step index)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# header line\n")
        for k in range(n_rows):
            f.write(f"{k} 0.0 0.0 0.0\n")


def test_check_complete_returns_command_when_no_com_file(tmp_path):
    """Missing com_mol.dat -> simulation is not complete -> command is preserved."""
    sm = _bare_manager()
    command = (1000.0, 1, 0, 0, str(tmp_path), 0)
    out = sm.check_complete(command, nsteps=10_000, nfreq=1000, nchains=10)
    assert out == command


def test_check_complete_returns_false_when_steps_reached(tmp_path):
    """com_mol.dat with enough rows -> nsteps_out >= nsteps -> returns False (drop)."""
    sm = _bare_manager()
    # nsteps_out = (rows / nchains) * nfreq -> need rows >= nsteps/nfreq * nchains
    nchains, nfreq, nsteps = 10, 1000, 10_000
    rows = (nsteps // nfreq) * nchains + 5  # slight margin

    com = tmp_path / "poly3" / "0" / "com_mol.dat"
    _write_com_mol(com, n_rows=rows)

    command = (1000.0, 1, 3, 0, str(tmp_path), 0)
    out = sm.check_complete(command, nsteps=nsteps, nfreq=nfreq, nchains=nchains)
    assert out is False


def test_check_complete_keeps_command_when_steps_insufficient(tmp_path):
    """Too few rows -> nsteps_out < nsteps -> keep command for re-run."""
    sm = _bare_manager()
    nchains, nfreq, nsteps = 10, 1000, 10_000
    rows = (nsteps // nfreq) * nchains // 2  # half the required rows

    com = tmp_path / "poly3" / "0" / "com_mol.dat"
    _write_com_mol(com, n_rows=rows)

    command = (1000.0, 1, 3, 0, str(tmp_path), 0)
    out = sm.check_complete(command, nsteps=nsteps, nfreq=nfreq, nchains=nchains)
    assert out == command


def test_check_complete_keeps_command_when_file_corrupt(tmp_path):
    """np.loadtxt failure is caught and the command is preserved (safe default)."""
    sm = _bare_manager()
    com = tmp_path / "poly3" / "0" / "com_mol.dat"
    com.parent.mkdir(parents=True, exist_ok=True)
    com.write_text("# this is garbage with no numeric data at all\n???\n!!!\n")

    command = (1000.0, 1, 3, 0, str(tmp_path), 0)
    out = sm.check_complete(command, nsteps=10_000, nfreq=1000, nchains=10)
    assert out == command
