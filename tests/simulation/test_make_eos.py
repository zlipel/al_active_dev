"""
Unit tests for the pure helpers in `simulation/make_eos.py`.

We cover `SimulationManager.filter_by_completion`, which decides which (poly, rho)
directories need to be (re-)run based on thermo.avg's last timestep. Ray is not
involved on this path; only filesystem state matters.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from simulation.make_eos import SimulationManager


def _bare_manager() -> SimulationManager:
    return SimulationManager.__new__(SimulationManager)


def _write_thermo(path: Path, last_timestep: int) -> None:
    """Write a minimal thermo.avg with a header and one numeric row at `last_timestep`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Time Temp Etot PE KE Ent P rho\n"
        "# units\n"
        f"{last_timestep} 300.0 0.0 0.0 0.0 0.0 0.5 0.5\n"
    )


def _cmd(i: int, density: float, parent: Path) -> tuple:
    return (1000.0, 1, i, density, str(parent))


def test_filter_keeps_when_output_dir_missing(tmp_path):
    """Directory absent -> simulation hasn't started -> keep command."""
    sm = _bare_manager()
    cmds = [_cmd(0, 0.4, tmp_path)]
    kept = sm.filter_by_completion(cmds, nsteps=10_000_000)
    assert kept == cmds


def test_filter_keeps_when_thermo_avg_missing(tmp_path):
    """Directory exists but no thermo.avg -> rerun -> keep command."""
    sm = _bare_manager()
    (tmp_path / "poly0" / "rho0.4").mkdir(parents=True)
    cmds = [_cmd(0, 0.4, tmp_path)]
    kept = sm.filter_by_completion(cmds, nsteps=10_000_000)
    assert kept == cmds


def test_filter_drops_when_last_timestep_meets_nsteps(tmp_path):
    """thermo.avg with last_timestep >= nsteps -> skip (drop from kept list)."""
    sm = _bare_manager()
    _write_thermo(tmp_path / "poly0" / "rho0.4" / "thermo.avg", last_timestep=10_000_000)
    cmds = [_cmd(0, 0.4, tmp_path)]
    kept = sm.filter_by_completion(cmds, nsteps=10_000_000)
    assert kept == []


def test_filter_keeps_when_last_timestep_below_nsteps(tmp_path):
    """thermo.avg present but simulation didn't reach nsteps -> restart from scratch."""
    sm = _bare_manager()
    _write_thermo(tmp_path / "poly0" / "rho0.4" / "thermo.avg", last_timestep=5_000_000)
    cmds = [_cmd(0, 0.4, tmp_path)]
    kept = sm.filter_by_completion(cmds, nsteps=10_000_000)
    assert kept == cmds


def test_filter_keeps_when_thermo_avg_corrupt(tmp_path):
    """Unparseable last line -> exception is caught -> command is preserved (safe)."""
    sm = _bare_manager()
    thermo = tmp_path / "poly0" / "rho0.4" / "thermo.avg"
    thermo.parent.mkdir(parents=True)
    thermo.write_text("# header\n# header2\nthis row is not numeric at all\n")
    cmds = [_cmd(0, 0.4, tmp_path)]
    kept = sm.filter_by_completion(cmds, nsteps=10_000_000)
    assert kept == cmds


def test_filter_mixed_batch(tmp_path):
    """Mix completed and incomplete simulations; only incomplete ones survive."""
    sm = _bare_manager()
    # poly0 / rho0.4 — completed
    _write_thermo(tmp_path / "poly0" / "rho0.4" / "thermo.avg", last_timestep=10_000_000)
    # poly1 / rho0.4 — incomplete
    _write_thermo(tmp_path / "poly1" / "rho0.4" / "thermo.avg", last_timestep=2_000_000)
    # poly2 / rho0.4 — no thermo.avg, dir exists
    (tmp_path / "poly2" / "rho0.4").mkdir(parents=True)
    # poly3 / rho0.4 — no dir at all
    cmds = [_cmd(i, 0.4, tmp_path) for i in range(4)]

    kept = sm.filter_by_completion(cmds, nsteps=10_000_000)
    kept_ids = sorted(c[2] for c in kept)
    assert kept_ids == [1, 2, 3]
