"""
Tests for the minimization-space conversion that underpins all EHVI math.

The whole acquisition module operates in MIN space:
  - 'upper' fronts (max-max) get negated on the way in
  - 'lower' fronts (min-min) pass through unchanged
The reference-point selection and the analytic stripe decomposition both rely
on this invariant, so we lock it down here.
"""
from __future__ import annotations

import numpy as np
import pytest

from al_pipeline.acquisition.ehvi import (
    _to_min_space,
    _get_ref_point,
    _ref_point_frac,
    _ref_point_on_IN_line,
    _ref_point_halfway,
    _augment_front,
    front_augmentation,
)


# ---------- _to_min_space ----------

def test_to_min_space_upper_negates():
    """upper front (max-max) -> negated so larger original = smaller min-space."""
    Y = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    out = _to_min_space(Y, front="upper")
    np.testing.assert_allclose(out, -Y)


def test_to_min_space_lower_is_identity():
    """lower front (min-min) passes through unchanged."""
    Y = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    out = _to_min_space(Y, front="lower")
    np.testing.assert_allclose(out, Y)


def test_to_min_space_invalid_front_raises():
    Y = np.array([[1.0, 2.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="front"):
        _to_min_space(Y, front="diagonal")


def test_to_min_space_preserves_shape():
    Y = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    assert _to_min_space(Y, "upper").shape == Y.shape
    assert _to_min_space(Y, "lower").shape == Y.shape


# ---------- reference point selection ----------

def test_ref_point_frac_is_worse_than_nadir():
    """frac ref point must dominate (in min-sense, be worse than) the nadir for HV
    to be well-defined."""
    pmin = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0]], dtype=np.float32)
    nadir = pmin.max(axis=0)
    rp = _ref_point_frac(pmin, frac=0.5)
    assert np.all(rp >= nadir)


def test_ref_point_in_line_dominates_nadir_strictly():
    """in_line ref point must be strictly worse than nadir (no ties)."""
    pmin = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0]], dtype=np.float32)
    nadir = pmin.max(axis=0)
    rp = _ref_point_on_IN_line(pmin, tau=0.05, cap_frac=0.8)
    assert np.all(rp > nadir)


def test_ref_point_halfway_between_frac_and_nadir():
    """halfway sits between the naive frac point and the nadir along the I->N axis."""
    pmin = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0]], dtype=np.float32)
    nadir = pmin.max(axis=0)
    rp_frac = _ref_point_frac(pmin, frac=0.5)
    rp_half = _ref_point_halfway(pmin, frac=0.5)
    # halfway must be no farther from nadir than frac is
    d_frac = np.linalg.norm(rp_frac - nadir)
    d_half = np.linalg.norm(rp_half - nadir)
    assert d_half <= d_frac + 1e-6


def test_get_ref_point_unknown_mode_raises():
    pmin = np.array([[1.0, 4.0], [2.0, 3.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="Unknown ref_mode"):
        _get_ref_point(pmin, ref_mode="bogus")


# ---------- _augment_front ----------

def test_augment_front_sorts_by_y2_ascending():
    """The middle rows (between sentinels) must be sorted by y2 ascending."""
    pf_min = np.array([[3.0, 1.0], [1.0, 3.0], [2.0, 2.0]], dtype=np.float32)
    ref = np.array([5.0, 5.0], dtype=np.float32)
    aug = _augment_front(pf_min, ref, big=1e6)

    middle = aug[1:-1]
    assert np.all(np.diff(middle[:, 1]) >= 0), "middle rows not sorted by y2 ascending"


def test_augment_front_sentinels_are_first_and_last():
    """Sentinel rows must bracket the front: (r1, -big) first, (-big, r2) last."""
    pf_min = np.array([[1.0, 3.0], [2.0, 2.0]], dtype=np.float32)
    ref = np.array([5.0, 5.0], dtype=np.float32)
    big = 1e6
    aug = _augment_front(pf_min, ref, big=big)

    assert aug[0, 0] == pytest.approx(5.0)
    assert aug[0, 1] == pytest.approx(-big)
    assert aug[-1, 0] == pytest.approx(-big)
    assert aug[-1, 1] == pytest.approx(5.0)


def test_augment_front_shape_is_N_plus_2():
    pf_min = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]], dtype=np.float32)
    ref = np.array([5.0, 5.0], dtype=np.float32)
    aug = _augment_front(pf_min, ref, big=1e6)
    assert aug.shape == (pf_min.shape[0] + 2, 2)


# ---------- front_augmentation (public API) ----------

def test_front_augmentation_upper_lower_are_negation_symmetric():
    """
    Augmenting an upper front of Y and a lower front of -Y should produce the same
    augmented front (since both end up in MIN space via the same code path).
    """
    Y = np.array([[1.0, 5.0], [3.0, 3.0], [5.0, 1.0]], dtype=np.float32)
    aug_upper = front_augmentation(Y, front="upper", ref_mode="frac", frac=0.5)
    aug_lower = front_augmentation(-Y, front="lower", ref_mode="frac", frac=0.5)
    np.testing.assert_allclose(aug_upper, aug_lower, atol=1e-6)


def test_front_augmentation_returns_ref_when_requested():
    Y = np.array([[1.0, 5.0], [3.0, 3.0]], dtype=np.float32)
    aug, ref = front_augmentation(Y, front="lower", return_ref=True)
    assert aug.shape == (Y.shape[0] + 2, 2)
    assert ref.shape == (2,)


def test_front_augmentation_mc_mode_skips_sentinels():
    """MC-EHVI uses the raw MIN-space points (no sentinels) — verify shape."""
    Y = np.array([[1.0, 5.0], [3.0, 3.0]], dtype=np.float32)
    aug = front_augmentation(Y, front="lower", mc_mode=True)
    assert aug.shape == Y.shape
