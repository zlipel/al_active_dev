"""
Closed-form sanity checks for the analytic stripe-decomposed EHVI.

Two properties we lean on:
  - As sigma -> 0, EHVI(mu, sigma) -> HV improvement at the mean (deterministic HVI).
  - EHVI is symmetric under a coordinated swap of the two objectives.

Both are derivable from the integral definition of EHVI and are useful guard rails
against indexing bugs in the stripe decomposition.
"""
from __future__ import annotations

import numpy as np
import pytest
from pygmo import hypervolume

from al_pipeline.acquisition.ehvi import (
    ehvi_analytic,
    front_augmentation,
    filter_nondominated,
    is_dominated_by_front,
)


def _deterministic_hvi(mu_min: np.ndarray, pf_min: np.ndarray, ref_min: np.ndarray) -> float:
    """HV improvement when the candidate is at mu deterministically (sigma -> 0)."""
    base_hv = hypervolume(pf_min).compute(ref_min)
    if is_dominated_by_front(pf_min, mu_min):
        return 0.0
    if np.any(mu_min >= ref_min):
        return 0.0
    extended = np.vstack([pf_min, mu_min])
    nd = extended[filter_nondominated(extended)]
    new_hv = hypervolume(nd).compute(ref_min)
    return max(0.0, new_hv - base_hv)


# ---------- shape / non-negativity ----------

def test_ehvi_returns_per_candidate_array():
    """Output shape is (n_candidates,) for a batch input."""
    pf = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]], dtype=np.float32)
    aug = front_augmentation(pf, front="lower", ref_mode="frac", frac=0.5)

    mus1 = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    mus2 = np.array([2.5, 1.5, 0.5], dtype=np.float32)
    sigmas = np.full(3, 0.3, dtype=np.float32)

    out = ehvi_analytic(mus1, sigmas, mus2, sigmas, aug)
    assert out.shape == (3,)


def test_ehvi_is_non_negative():
    """EHVI is an expected improvement; it must be >= 0 by construction."""
    rng = np.random.default_rng(0)
    pf = np.array([[1.0, 4.0], [2.0, 2.0], [4.0, 1.0]], dtype=np.float32)
    aug = front_augmentation(pf, front="lower", ref_mode="frac", frac=0.5)

    mus1 = rng.uniform(-1.0, 6.0, size=20).astype(np.float32)
    mus2 = rng.uniform(-1.0, 6.0, size=20).astype(np.float32)
    sigmas1 = rng.uniform(0.05, 1.0, size=20).astype(np.float32)
    sigmas2 = rng.uniform(0.05, 1.0, size=20).astype(np.float32)

    out = ehvi_analytic(mus1, sigmas1, mus2, sigmas2, aug)
    assert np.all(out >= -1e-6), f"min EHVI was {out.min()}"


# ---------- low-sigma limit ----------

@pytest.mark.parametrize("mu", [(0.0, 0.0), (0.5, 0.5), (1.5, 2.5), (2.5, 1.5)])
def test_ehvi_small_sigma_matches_deterministic_hvi(mu):
    """For tiny sigma, analytic EHVI must agree with the deterministic HV improvement."""
    pf = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]], dtype=np.float32)
    aug, ref = front_augmentation(pf, front="lower", ref_mode="frac", frac=0.5, return_ref=True)

    mu_arr = np.array([mu], dtype=np.float32)
    sigma = np.array([1e-3], dtype=np.float32)

    ehvi = float(
        ehvi_analytic(mu_arr[:, 0], sigma, mu_arr[:, 1], sigma, aug)[0]
    )
    hvi = _deterministic_hvi(np.array(mu, dtype=np.float32), pf, ref)
    # tolerance reflects the residual sigma; tighten if we go to 1e-5 sigma
    assert abs(ehvi - hvi) < 5e-2, f"ehvi={ehvi}, hvi={hvi}"


def test_ehvi_dominated_candidate_has_small_value():
    """A point far inside the dominated region (and inside ref) yields ~0 EHVI."""
    pf = np.array([[0.0, 0.0]], dtype=np.float32)  # ideal point at origin
    aug = front_augmentation(pf, front="lower", ref_mode="frac", frac=0.5)

    # candidate way out near the ref, small sigma -> almost certainly dominated
    mu1 = np.array([0.49], dtype=np.float32)
    mu2 = np.array([0.49], dtype=np.float32)
    sigma = np.array([1e-3], dtype=np.float32)

    out = float(ehvi_analytic(mu1, sigma, mu2, sigma, aug)[0])
    assert out < 1e-3, f"expected ~0 EHVI for deeply dominated candidate, got {out}"


# ---------- symmetry under axis swap ----------

def test_ehvi_symmetric_under_objective_swap():
    """Swapping obj1<->obj2 (front, candidate, sigmas all coordinated) preserves EHVI."""
    pf = np.array([[1.0, 4.0], [2.0, 2.0], [4.0, 1.0]], dtype=np.float32)
    aug = front_augmentation(pf, front="lower", ref_mode="frac", frac=0.5)

    mu1 = np.array([1.5, 0.5, 3.0], dtype=np.float32)
    mu2 = np.array([2.5, 3.5, 1.5], dtype=np.float32)
    s1  = np.array([0.4, 0.3, 0.5], dtype=np.float32)
    s2  = np.array([0.5, 0.6, 0.3], dtype=np.float32)

    e_orig = ehvi_analytic(mu1, s1, mu2, s2, aug)

    pf_sw = pf[:, ::-1]
    aug_sw = front_augmentation(pf_sw, front="lower", ref_mode="frac", frac=0.5)
    e_swap = ehvi_analytic(mu2, s2, mu1, s1, aug_sw)

    np.testing.assert_allclose(e_orig, e_swap, atol=1e-5)


# ---------- monotonicity in posterior mean ----------

def test_ehvi_monotone_in_mean_for_non_dominated_region():
    """
    Moving the candidate mean toward the ideal (smaller in both dims) — while staying
    non-dominated and outside the current front's dominated rectangle — should not
    decrease EHVI.
    """
    pf = np.array([[2.0, 2.0]], dtype=np.float32)
    aug = front_augmentation(pf, front="lower", ref_mode="frac", frac=0.5)

    sigma = np.array([0.1], dtype=np.float32)

    e_far  = float(ehvi_analytic(np.array([1.5]), sigma, np.array([1.5]), sigma, aug)[0])
    e_near = float(ehvi_analytic(np.array([0.5]), sigma, np.array([0.5]), sigma, aug)[0])

    assert e_near >= e_far - 1e-6, f"e_near={e_near}, e_far={e_far}"


def test_ehvi_zero_at_sentinel_extremes():
    """A candidate at the very-bad sentinel rectangle corner gives ~0 EHVI."""
    pf = np.array([[1.0, 1.0]], dtype=np.float32)
    aug, ref = front_augmentation(pf, front="lower", ref_mode="frac", frac=0.5, return_ref=True)

    # Place candidate strictly worse than ref in both dims -> can't improve HV.
    mu1 = np.array([float(ref[0]) + 0.5], dtype=np.float32)
    mu2 = np.array([float(ref[1]) + 0.5], dtype=np.float32)
    sigma = np.array([1e-3], dtype=np.float32)

    out = float(ehvi_analytic(mu1, sigma, mu2, sigma, aug)[0])
    assert out < 1e-3
