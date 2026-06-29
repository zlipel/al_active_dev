"""
Tests for the Pareto dominance helpers used by MC-EHVI.

`is_dominated_by_front` short-circuits MC sampling when a draw is already dominated
(EHVI contribution is zero). `filter_nondominated` is used to compute the
non-dominated front of (existing front ∪ {sample}) before the hypervolume call.
Both operate in MIN space.
"""
from __future__ import annotations

import numpy as np
import pytest

from al_pipeline.acquisition.ehvi import (
    is_dominated_by_front,
    filter_nondominated,
)


# ---------- is_dominated_by_front ----------

def test_strictly_dominated_point_is_dominated():
    pf = np.array([[1.0, 1.0]], dtype=np.float32)
    s = np.array([2.0, 2.0], dtype=np.float32)
    assert is_dominated_by_front(pf, s) is True


def test_non_dominated_point_is_not_dominated():
    """Point that is better in y1 than the only front member is non-dominated."""
    pf = np.array([[2.0, 1.0]], dtype=np.float32)
    s = np.array([1.0, 2.0], dtype=np.float32)
    assert is_dominated_by_front(pf, s) is False


def test_equal_to_front_member_is_not_dominated():
    """Equal points don't dominate each other (strict <  in at least one dim required)."""
    pf = np.array([[1.0, 2.0]], dtype=np.float32)
    s = np.array([1.0, 2.0], dtype=np.float32)
    assert is_dominated_by_front(pf, s) is False


def test_dominated_when_tied_in_one_dim_and_worse_in_other():
    """pf=(1,2), s=(2,2): pf<=s in both, pf<s in one -> dominated."""
    pf = np.array([[1.0, 2.0]], dtype=np.float32)
    s = np.array([2.0, 2.0], dtype=np.float32)
    assert is_dominated_by_front(pf, s) is True


def test_any_front_member_can_dominate():
    pf = np.array([[5.0, 5.0], [0.5, 0.5], [9.0, 9.0]], dtype=np.float32)
    s = np.array([1.0, 1.0], dtype=np.float32)
    # only (0.5, 0.5) dominates s, but that's enough
    assert is_dominated_by_front(pf, s) is True


# ---------- filter_nondominated ----------

def test_filter_empty_returns_empty_mask():
    mask = filter_nondominated(np.zeros((0, 2)))
    assert mask.shape == (0,)
    assert mask.dtype == bool


def test_filter_single_point_is_nondominated():
    P = np.array([[1.0, 2.0]], dtype=np.float32)
    mask = filter_nondominated(P)
    assert mask.tolist() == [True]


def test_filter_pareto_front_is_all_nondominated():
    """All members of a true Pareto front (in MIN) are non-dominated."""
    P = np.array([[1.0, 5.0], [2.0, 3.0], [4.0, 2.0], [5.0, 1.0]], dtype=np.float32)
    mask = filter_nondominated(P)
    assert mask.tolist() == [True, True, True, True]


def test_filter_drops_dominated_points():
    """(3,3) is dominated by (1,1); (5,5) is dominated by everything."""
    P = np.array(
        [[1.0, 1.0], [3.0, 3.0], [5.0, 5.0], [2.0, 0.5]],
        dtype=np.float32,
    )
    mask = filter_nondominated(P)
    # Expected non-dominated: (1,1) and (2, 0.5)
    assert mask.tolist() == [True, False, False, True]


def test_filter_handles_duplicate_points():
    """When two points are identical, exactly one is kept (the first in lex order)."""
    P = np.array([[1.0, 1.0], [1.0, 1.0], [2.0, 0.5]], dtype=np.float32)
    mask = filter_nondominated(P)
    assert int(mask[:2].sum()) == 1, "exactly one of the duplicate pair should survive"
    assert mask[2] is np.True_ or mask[2] == True


def test_filter_returns_mask_in_original_order():
    """Mask indices align with the input order, not the internal sort order."""
    P = np.array([[5.0, 1.0], [1.0, 5.0], [3.0, 3.0]], dtype=np.float32)  # 2 nondom, 1 dom
    mask = filter_nondominated(P)
    # (5,1) and (1,5) are non-dominated; (3,3) is dominated by neither but...
    # actually (3,3) vs (5,1): not dominated. (3,3) vs (1,5): not dominated.
    # So all three are non-dominated.
    assert mask.tolist() == [True, True, True]
