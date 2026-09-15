"""Tests for the atomic per-endpoint checkpoint + resume helpers in
``beam_search/run_beams_mpi.py``.

Covers the pure result-assembly/write path, exercised without MPI or a real
bundle:

- ``assemble_results_df`` — grid order, rows_out overrides existing on shared
  keys, uncomputed endpoints omitted (missing key -> pending on resume).
- ``_atomic_write_csv`` — leaves a valid file and no ``.tmp`` residue.
- Mid-start durability: a partial ``paths.csv`` (some endpoints done, one still
  pending) leaves the start pending under ``get_pending_start_indices`` and is
  skipped once every endpoint has a final reason.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_BEAM_DIR = Path(__file__).resolve().parents[2] / "beam_search"
if str(_BEAM_DIR) not in sys.path:
    sys.path.insert(0, str(_BEAM_DIR))

from run_beams_mpi import (  # noqa: E402
    KEY_COLS,
    assemble_results_df,
    build_existing_reason_map,
    get_pending_start_indices,
    log_completion_summary,
    _atomic_write_csv,
)


def _order_df(start_idx, keys):
    """Canonical grid (start_idx, du_req, dv_req) + _order, as doWork builds it."""
    df = pd.DataFrame(
        [(start_idx, du, dv) for du, dv in keys], columns=KEY_COLS
    ).drop_duplicates().copy()
    df["_order"] = np.arange(len(df))
    return df


def _row(start_idx, du, dv, reason):
    return {"start_idx": start_idx, "du_req": du, "dv_req": dv, "reason": reason}


# ---------------------------------------------------------------------------
# assemble_results_df
# ---------------------------------------------------------------------------


def test_assemble_new_rows_win_and_grid_is_complete():
    keys = [(0.01, 0.0), (0.0, -0.02), (0.03, 0.03)]
    order_df = _order_df(7, keys)

    existing = pd.DataFrame([
        _row(7, 0.01, 0.0, "no_finished"),       # will be recomputed this session
        _row(7, 0.0, -0.02, "finished_quantile"),  # carried forward untouched
    ])
    rows_out = [
        _row(7, 0.01, 0.0, "finished_quantile"),   # new value wins
        _row(7, 0.03, 0.03, "outside_hull"),
    ]

    out = assemble_results_df(rows_out, existing, order_df)

    # exactly one row per grid cell, in grid order
    assert list(zip(out.du_req, out.dv_req)) == keys
    reason_by_key = {(r.du_req, r.dv_req): r.reason for r in out.itertuples(index=False)}
    assert reason_by_key[(0.01, 0.0)] == "finished_quantile"   # new won
    assert reason_by_key[(0.0, -0.02)] == "finished_quantile"  # existing kept
    assert reason_by_key[(0.03, 0.03)] == "outside_hull"


def test_assemble_uncomputed_endpoints_omitted():
    keys = [(0.01, 0.0), (0.0, -0.02), (0.03, 0.03)]
    order_df = _order_df(3, keys)

    rows_out = [
        _row(3, 0.01, 0.0, "finished_quantile"),
        _row(3, 0.0, -0.02, "outside_hull"),
    ]  # third endpoint not computed yet

    out = assemble_results_df(rows_out, existing_df=None, order_df=order_df)

    # Only computed endpoints are written; the third is absent, not NaN.
    assert list(zip(out.du_req, out.dv_req)) == [(0.01, 0.0), (0.0, -0.02)]
    reason_map = build_existing_reason_map(out)
    assert reason_map.get((3, 0.03, 0.03)) is None  # missing key -> pending


def test_blank_reason_reads_as_pending(tmp_path):
    # A legacy full-grid CSV with a NaN reason must resume, not be skipped.
    out_csv = tmp_path / "RESULTS" / "start_0000" / "paths.csv"
    out_csv.parent.mkdir(parents=True)
    df = pd.DataFrame([
        _row(0, 0.01, 0.0, "finished_quantile"),
        _row(0, 0.0, -0.02, np.nan),   # blank reason -> pending
    ])
    df.to_csv(out_csv, index=False)

    reason_map = build_existing_reason_map(pd.read_csv(out_csv))
    assert reason_map.get((0, 0.0, -0.02)) is None
    assert reason_map[(0, 0.01, 0.0)] == "finished_quantile"

    order_df = _order_df(0, [(0.01, 0.0), (0.0, -0.02)])
    pending, _, n_skipped = get_pending_start_indices(
        [0], {0: order_df[KEY_COLS].copy()}, str(tmp_path),
        resume=True, extend_no_finished=False,
    )
    assert pending == [0] and n_skipped == 0


def test_assemble_drops_keys_off_current_grid():
    keys = [(0.01, 0.0)]
    order_df = _order_df(1, keys)
    existing = pd.DataFrame([
        _row(1, 0.01, 0.0, "finished_quantile"),
        _row(1, 0.99, 0.99, "finished_quantile"),  # stale key, not in grid
    ])
    out = assemble_results_df([], existing, order_df)
    assert list(zip(out.du_req, out.dv_req)) == keys


# ---------------------------------------------------------------------------
# _atomic_write_csv
# ---------------------------------------------------------------------------


def test_atomic_write_leaves_valid_file_no_tmp(tmp_path):
    out_csv = tmp_path / "RESULTS" / "start_0007" / "paths.csv"
    out_csv.parent.mkdir(parents=True)
    df = pd.DataFrame([_row(7, 0.01, 0.0, "finished_quantile")])

    _atomic_write_csv(df, str(out_csv))
    back = pd.read_csv(out_csv)
    assert list(back.reason) == ["finished_quantile"]
    assert list(out_csv.parent.glob("*.tmp*")) == []

    # overwrite in place, still no residue
    _atomic_write_csv(pd.DataFrame([_row(7, 0.01, 0.0, "no_finished")]), str(out_csv))
    assert list(pd.read_csv(out_csv).reason) == ["no_finished"]
    assert list(out_csv.parent.glob("*.tmp*")) == []


# ---------------------------------------------------------------------------
# mid-start resume: a partial flush keeps the start pending
# ---------------------------------------------------------------------------


def test_partial_start_stays_pending_then_skipped(tmp_path):
    keys = [(0.01, 0.0), (0.0, -0.02), (0.03, 0.03)]
    order_df = _order_df(5, keys)
    groups_by_start = {5: order_df[KEY_COLS].copy()}
    paths_dir = str(tmp_path)
    out_csv = tmp_path / "RESULTS" / "start_0005" / "paths.csv"
    out_csv.parent.mkdir(parents=True)

    # Partial flush: two endpoints done, third not yet computed (absent).
    partial = assemble_results_df(
        [_row(5, 0.01, 0.0, "finished_quantile"), _row(5, 0.0, -0.02, "outside_hull")],
        existing_df=None, order_df=order_df,
    )
    _atomic_write_csv(partial, str(out_csv))

    pending, n_with_results, n_skipped = get_pending_start_indices(
        [5], groups_by_start, paths_dir, resume=True, extend_no_finished=False,
    )
    assert pending == [5] and n_with_results == 1 and n_skipped == 0

    # Worker-side invariant (the doWork keep_mask predicate): after a real CSV
    # round-trip the absent endpoint reads as None (retried), done ones as final.
    from run_beams_mpi import FINAL_REASONS
    reason_map = build_existing_reason_map(pd.read_csv(out_csv))
    assert reason_map.get((5, 0.03, 0.03)) is None
    assert reason_map[(5, 0.01, 0.0)] in FINAL_REASONS
    assert reason_map[(5, 0.0, -0.02)] in FINAL_REASONS

    # Complete the start; now it is skipped on resume.
    full = assemble_results_df(
        [_row(5, 0.03, 0.03, "finished_quantile")],
        existing_df=partial, order_df=order_df,
    )
    _atomic_write_csv(full, str(out_csv))

    pending, _, n_skipped = get_pending_start_indices(
        [5], groups_by_start, paths_dir, resume=True, extend_no_finished=False,
    )
    assert pending == [] and n_skipped == 1


def _write_paths(paths_dir, start_idx, rows):
    d = paths_dir / "RESULTS" / f"start_{start_idx:04d}"
    d.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(d / "paths.csv", index=False)


def test_completion_summary_flags_incomplete(tmp_path):
    o1 = _order_df(1, [(0.01, 0.0), (0.0, -0.02)])
    o2 = _order_df(2, [(0.01, 0.0), (0.0, -0.02)])
    groups = {1: o1[KEY_COLS].copy(), 2: o2[KEY_COLS].copy()}
    # start 1 complete (both endpoints final); start 2 missing its 2nd endpoint.
    _write_paths(tmp_path, 1, [_row(1, 0.01, 0.0, "finished_quantile"),
                               _row(1, 0.0, -0.02, "no_finished")])
    _write_paths(tmp_path, 2, [_row(2, 0.01, 0.0, "finished_quantile")])

    incomplete = log_completion_summary([1, 2], groups, str(tmp_path))
    assert incomplete == [2]


def test_completion_summary_all_complete(tmp_path):
    o1 = _order_df(1, [(0.01, 0.0)])
    groups = {1: o1[KEY_COLS].copy()}
    _write_paths(tmp_path, 1, [_row(1, 0.01, 0.0, "finished_quantile")])

    assert log_completion_summary([1], groups, str(tmp_path)) == []


def test_completion_summary_missing_file_is_incomplete(tmp_path):
    # A start whose worker never wrote paths.csv (e.g. killed) is incomplete.
    o1 = _order_df(9, [(0.01, 0.0)])
    groups = {9: o1[KEY_COLS].copy()}
    assert log_completion_summary([9], groups, str(tmp_path)) == [9]
