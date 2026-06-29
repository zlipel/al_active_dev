"""
Tests for `generate_simulation_candidates`, the bridge between al_pipeline and the
simulation submit scripts.

What we lock:
  - The output file is written at `paths.next_iter_candidates_file` with one
    sequence per line in the order seq_child_1 .. seq_child_ngen.
  - The same file is mirrored into both `next_iter_eos_dir` and `next_iter_diff_dir`
    with the same name (this is what `make_eos.sh` reads from EOS/).
  - Missing or empty child files raise rather than silently producing a partial file.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from al_pipeline.core.config import ALConfig
from al_pipeline.core.paths import ensure_dirs
from al_pipeline.data_prep.simulation_candidates import generate_simulation_candidates


def _make_cfg(tmp_path: Path, *, ngen: int) -> ALConfig:
    return ALConfig(
        model="hps_urry",
        iteration=0,
        front="upper",
        ngen=ngen,
        base_path=tmp_path / "home",
        scratch_path=tmp_path / "scratch",
    )


def _populate_children(cfg: ALConfig, seqs: list[str]) -> None:
    """Write the expected seq_child_{i}.txt files for `seqs`."""
    p = cfg.paths
    p.ga_children_dir.mkdir(parents=True, exist_ok=True)
    for i, seq in enumerate(seqs, start=1):
        (p.ga_children_dir / f"seq_child_{i}.txt").write_text(seq + "\n")


def test_writes_one_sequence_per_line_in_order(tmp_path):
    cfg = _make_cfg(tmp_path, ngen=4)
    ensure_dirs(cfg.paths)
    seqs = ["AAAA", "GGGG", "DEKR", "GSGS"]
    _populate_children(cfg, seqs)

    out = generate_simulation_candidates(cfg)

    lines = out.read_text().strip().splitlines()
    assert lines == seqs


def test_writes_to_next_iter_candidates_file(tmp_path):
    cfg = _make_cfg(tmp_path, ngen=2)
    ensure_dirs(cfg.paths)
    _populate_children(cfg, ["AAAA", "GGGG"])

    out = generate_simulation_candidates(cfg)
    assert out == cfg.paths.next_iter_candidates_file


def test_mirrors_into_eos_and_diff_dirs(tmp_path):
    """make_eos.sh expects this file under SIMULATIONS/EOS/, so the mirror must happen."""
    cfg = _make_cfg(tmp_path, ngen=3)
    ensure_dirs(cfg.paths)
    seqs = ["AAAA", "GGGG", "DEKR"]
    _populate_children(cfg, seqs)

    out = generate_simulation_candidates(cfg)
    name = out.name

    eos_copy = cfg.paths.next_iter_eos_dir / name
    diff_copy = cfg.paths.next_iter_diff_dir / name

    assert eos_copy.exists()
    assert diff_copy.exists()
    assert eos_copy.read_text() == out.read_text()
    assert diff_copy.read_text() == out.read_text()


def test_missing_child_file_raises(tmp_path):
    cfg = _make_cfg(tmp_path, ngen=3)
    ensure_dirs(cfg.paths)
    _populate_children(cfg, ["AAAA", "GGGG"])  # only 2 of 3

    with pytest.raises(FileNotFoundError, match="Missing child file"):
        generate_simulation_candidates(cfg)


def test_empty_sequence_in_child_raises(tmp_path):
    cfg = _make_cfg(tmp_path, ngen=2)
    ensure_dirs(cfg.paths)
    (cfg.paths.ga_children_dir / "seq_child_1.txt").write_text("AAAA\n")
    (cfg.paths.ga_children_dir / "seq_child_2.txt").write_text("\n")  # empty

    with pytest.raises(ValueError, match="Empty sequence"):
        generate_simulation_candidates(cfg)


def test_missing_children_dir_raises(tmp_path):
    cfg = _make_cfg(tmp_path, ngen=2)
    # do NOT create ga_children_dir
    with pytest.raises(FileNotFoundError, match="Missing children dir"):
        generate_simulation_candidates(cfg)
