"""Unit tests for the pure helpers in ``beam_search/prepare_endpoints.py``.

Covers the four functions that decide start selection + target grid, without
loading a real bundle:

- ``stratify_pool_by_grid``
- ``farthest_point_bin_sample``
- ``select_benchmark_starts``
- ``select_production_starts``
- ``make_target_deltas``
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Runner-side scripts use unqualified imports (e.g., ``from cross_paths...``)
# because they're invoked with ``beam_search/`` on PYTHONPATH by the SLURM
# submit scripts. Add it here so pure-Python helpers in prepare_endpoints
# are importable from the test root.
_BEAM_DIR = Path(__file__).resolve().parents[2] / "beam_search"
if str(_BEAM_DIR) not in sys.path:
    sys.path.insert(0, str(_BEAM_DIR))

from prepare_endpoints import (  # noqa: E402
    farthest_point_bin_sample,
    filter_reachable_starts,
    make_target_deltas,
    select_benchmark_starts,
    select_production_starts,
    stratify_pool_by_grid,
)


# ---------------------------------------------------------------------------
# stratify_pool_by_grid
# ---------------------------------------------------------------------------


def test_stratify_bins_are_disjoint_and_cover_pool():
    rng = np.random.default_rng(0)
    uv = rng.uniform(0.0, 1.0, size=(200, 2))
    pool_idx = np.arange(200)

    bins = stratify_pool_by_grid(uv, pool_idx, n_bins=4)

    # Every pool index appears in exactly one bin.
    seen: set[int] = set()
    for members in bins.values():
        assert seen.isdisjoint(members.tolist())
        seen.update(int(i) for i in members)
    assert seen == set(pool_idx.tolist())


def test_stratify_upper_edge_lands_in_last_bin():
    # A sequence at exactly (u, v) = (1.0, 1.0) must land in bin (n-1, n-1)
    # rather than falling through the clip.
    uv = np.array([[1.0, 1.0], [0.0, 0.0], [0.5, 0.5]], dtype=float)
    bins = stratify_pool_by_grid(uv, np.arange(3), n_bins=4)
    assert (3, 3) in bins
    assert 0 in bins[(3, 3)].tolist()


def test_stratify_empty_pool_returns_empty_dict():
    uv = np.zeros((0, 2))
    bins = stratify_pool_by_grid(uv, np.array([], dtype=int), n_bins=3)
    assert bins == {}


def test_stratify_invalid_n_bins_raises():
    uv = np.zeros((2, 2))
    with pytest.raises(ValueError, match="positive"):
        stratify_pool_by_grid(uv, np.array([0, 1]), n_bins=0)


# ---------------------------------------------------------------------------
# farthest_point_bin_sample
# ---------------------------------------------------------------------------


def test_farthest_point_picks_opposite_corner_at_N2():
    corners = [(0, 0), (0, 4), (4, 0), (4, 4), (2, 2)]
    picked = farthest_point_bin_sample(corners, N=2)

    assert len(picked) == 2
    # First pick is (0, 0) (corner-closest by i+j). Second must be one of
    # the three bins at max Chebyshev distance 4 from (0, 0) — the exact
    # choice among tied bins is a stable-but-arbitrary lex tie-breaker.
    assert picked[0] == (0, 0)
    assert picked[1] in {(0, 4), (4, 0), (4, 4)}


def test_farthest_point_picks_four_extreme_corners_at_N4():
    # With N=4 on a 4-corner grid + a center point, the picked set should
    # cover all four corners regardless of the tie-breaking order.
    corners = [(0, 0), (0, 4), (4, 0), (4, 4), (2, 2)]
    picked = farthest_point_bin_sample(corners, N=4)
    assert set(picked) == {(0, 0), (0, 4), (4, 0), (4, 4)}


def test_farthest_point_wraps_when_N_exceeds_bin_count():
    bins = [(0, 0), (1, 1)]
    picked = farthest_point_bin_sample(bins, N=5)

    assert len(picked) == 5
    # Duplicates allowed only after every distinct bin has been picked once.
    assert set(picked[:2]) == set(bins)


def test_farthest_point_empty_returns_empty():
    assert farthest_point_bin_sample([], N=3) == []
    assert farthest_point_bin_sample([(0, 0)], N=0) == []


# ---------------------------------------------------------------------------
# select_benchmark_starts
# ---------------------------------------------------------------------------


def test_benchmark_picks_one_per_bin_at_small_N():
    bins = {
        (0, 0): np.array([10, 11, 12]),
        (0, 3): np.array([20]),
        (3, 0): np.array([30, 31]),
        (3, 3): np.array([40, 41, 42, 43]),
    }
    rng = np.random.default_rng(0)
    picks = select_benchmark_starts(bins, N=4, rng=rng)

    assert len(picks) == 4
    picked_bins = [b for b, _ in picks]
    # Every non-empty bin represented exactly once.
    assert sorted(picked_bins) == sorted(bins.keys())

    for bkey, seq_idx in picks:
        assert seq_idx in bins[bkey].tolist()


def test_benchmark_deterministic_with_seed():
    bins = {
        (0, 0): np.arange(10),
        (2, 2): np.arange(10, 20),
        (4, 4): np.arange(20, 30),
    }
    picks_a = select_benchmark_starts(bins, N=3, rng=np.random.default_rng(7))
    picks_b = select_benchmark_starts(bins, N=3, rng=np.random.default_rng(7))
    assert picks_a == picks_b


def test_benchmark_zero_N_returns_empty():
    bins = {(0, 0): np.array([1, 2])}
    assert select_benchmark_starts(bins, N=0, rng=np.random.default_rng(0)) == []
    assert select_benchmark_starts({}, N=3, rng=np.random.default_rng(0)) == []


# ---------------------------------------------------------------------------
# select_production_starts
# ---------------------------------------------------------------------------


def test_production_takes_ceil_frac_per_bin():
    bins = {
        (0, 0): np.arange(10),   # frac=0.5 → ceil(5) = 5
        (1, 1): np.arange(20),   # frac=0.5 → 10
        (2, 2): np.array([100]), # frac=0.5 → ceil(0.5) = 1 (all)
    }
    picks = select_production_starts(bins, frac=0.5, rng=np.random.default_rng(0))

    per_bin: dict[tuple[int, int], int] = {}
    for bkey, _ in picks:
        per_bin[bkey] = per_bin.get(bkey, 0) + 1
    assert per_bin[(0, 0)] == 5
    assert per_bin[(1, 1)] == 10
    assert per_bin[(2, 2)] == 1


def test_production_frac_one_picks_every_sequence():
    bins = {
        (0, 0): np.array([1, 2, 3]),
        (1, 1): np.array([4, 5]),
    }
    picks = select_production_starts(bins, frac=1.0, rng=np.random.default_rng(0))
    all_picked = sorted(seq for _, seq in picks)
    assert all_picked == [1, 2, 3, 4, 5]


def test_production_no_replacement_within_bin():
    bins = {(0, 0): np.arange(5)}
    picks = select_production_starts(bins, frac=1.0, rng=np.random.default_rng(3))
    seqs = [seq for _, seq in picks]
    assert len(seqs) == len(set(seqs))


# ---------------------------------------------------------------------------
# make_target_deltas
# ---------------------------------------------------------------------------


def test_benchmark_deltas_are_four_diagonals():
    dels = make_target_deltas(
        "benchmark",
        grid_spacing=0.0125,
        largest_delta=0.05,
        benchmark_delta=0.0375,
    )
    assert set(dels) == {(0.0375, 0.0375), (0.0375, -0.0375),
                        (-0.0375, 0.0375), (-0.0375, -0.0375)}


def test_production_deltas_are_8x8_no_zero():
    dels = make_target_deltas(
        "production",
        grid_spacing=0.0125,
        largest_delta=0.05,
        benchmark_delta=0.0375,
    )
    assert len(dels) == 64
    # No cell has du=0 or dv=0.
    assert all(abs(du) > 1e-9 and abs(dv) > 1e-9 for du, dv in dels)
    # Symmetric on both axes.
    dus = sorted({du for du, _ in dels})
    dvs = sorted({dv for _, dv in dels})
    assert dus == dvs
    assert dus == sorted({-round(0.0125 * (i + 1), 6) for i in range(4)}
                         | {round(0.0125 * (i + 1), 6) for i in range(4)})


def test_production_default_smaller_grid():
    # spacing=0.02, largest=0.06 → 3 pos + 3 neg per axis → 6×6 = 36
    dels = make_target_deltas(
        "production",
        grid_spacing=0.02,
        largest_delta=0.06,
        benchmark_delta=0.0375,
    )
    assert len(dels) == 36


def test_make_target_deltas_bad_mode_raises():
    with pytest.raises(ValueError, match="unknown mode"):
        make_target_deltas("junk", grid_spacing=0.01, largest_delta=0.05,
                          benchmark_delta=0.02)


def test_make_target_deltas_production_zero_grid_raises():
    with pytest.raises(ValueError, match="positive"):
        make_target_deltas("production", grid_spacing=0.0, largest_delta=0.05,
                          benchmark_delta=0.02)


def test_make_target_deltas_production_step_larger_than_bound_raises():
    with pytest.raises(ValueError, match="no cells"):
        make_target_deltas("production", grid_spacing=0.1, largest_delta=0.05,
                          benchmark_delta=0.02)


# ---------------------------------------------------------------------------
# filter_reachable_starts — benchmark-mode reachability pre-filter
# ---------------------------------------------------------------------------

from shapely.geometry import MultiPoint  # noqa: E402


def _unit_square_hull():
    """Convex hull of the unit square — every (u, v) ∈ [0, 1]² is inside."""
    return MultiPoint([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]).convex_hull


def test_reachable_filter_keeps_interior_points():
    """A start with room to move ±delta on both axes without leaving the
    hull survives the filter."""
    hull = _unit_square_hull()
    # Interior points at least delta from all four hull edges.
    uv = np.array([[0.5, 0.5], [0.3, 0.7], [0.7, 0.3]], dtype=float)
    pool = np.arange(3)
    kept = filter_reachable_starts(pool, uv, hull, delta=0.1)
    assert sorted(kept.tolist()) == [0, 1, 2]


def test_reachable_filter_drops_corner_points():
    """Extreme-corner starts push at least one diagonal past the hull
    edge (via clipping) and are dropped."""
    hull = _unit_square_hull()
    # Add a labeled cluster in the middle so hull is smaller than [0, 1]²
    # for a more realistic test.
    inner_hull = MultiPoint([(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]).convex_hull
    uv = np.array([
        [0.5, 0.5],   # deep interior — reachable
        [0.9, 0.1],   # corner-adjacent — 3 of 4 diagonals push outside
        [0.15, 0.5],  # near left edge — (-, ±) diagonals clip
    ], dtype=float)
    pool = np.arange(3)
    kept = filter_reachable_starts(pool, uv, inner_hull, delta=0.1)
    assert 0 in kept.tolist()
    assert 1 not in kept.tolist()
    assert 2 not in kept.tolist()


def test_reachable_filter_empty_pool():
    hull = _unit_square_hull()
    kept = filter_reachable_starts(np.array([], dtype=int), np.zeros((0, 2)), hull, delta=0.1)
    assert kept.tolist() == []


def test_reachable_filter_respects_delta():
    """Same starts, tighter delta → more survive. Loose delta → more dropped."""
    hull = _unit_square_hull()
    uv = np.array([[0.05, 0.5], [0.5, 0.5], [0.95, 0.5]], dtype=float)
    pool = np.arange(3)
    # Δ=0.02: all three interior enough to stay in-hull after clipping.
    kept_small = filter_reachable_starts(pool, uv, hull, delta=0.02)
    # Δ=0.1: the two near-edge starts clip.
    kept_big = filter_reachable_starts(pool, uv, hull, delta=0.1)
    assert len(kept_small) >= len(kept_big)
    assert 1 in kept_big.tolist()  # centre point always survives
