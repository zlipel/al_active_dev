"""
Unit tests for the Monte Carlo EHVI helpers (`ehvi_samples` and `monte_carlo_ehvi_batch`).

`ehvi_samples` is the per-sample EHVI kernel and has three behaviors we want to lock:
  - dominated sample -> 0
  - sample beyond ref point -> 0
  - improving sample -> HV of (front ∪ {sample}) minus base HV

`monte_carlo_ehvi_batch` glues a posterior sampler to that kernel. We test it via a
fake `PoolPosterior` with fixed mean/std so the test does not need a fitted GP.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from al_pipeline.acquisition.ehvi import (
    ehvi_samples,
    monte_carlo_ehvi_batch,
    front_augmentation,
)
from al_pipeline.surrogates.base import PoolPosterior


# ---------- a PoolPosterior stub for monte_carlo_ehvi_batch ----------

class _FakePoolPosterior(PoolPosterior):
    """PoolPosterior with fixed per-candidate means / stds; sample() draws Gaussians.

    Mirrors what GlobalGPRSurrogate would return for a tiny analytical case so
    `monte_carlo_ehvi_batch` can be unit-tested without a fitted GP.
    """
    def __init__(self, mus: np.ndarray, sigmas: np.ndarray):
        self._means = np.asarray(mus, dtype=np.float64)
        self._stds = np.asarray(sigmas, dtype=np.float64)
        self._mus_t = torch.as_tensor(mus, dtype=torch.float32)
        self._sigmas_t = torch.as_tensor(sigmas, dtype=torch.float32)

    @property
    def means(self) -> np.ndarray:
        return self._means

    @property
    def stds(self) -> np.ndarray:
        return self._stds

    @property
    def covariance(self) -> np.ndarray:
        # Independent per-objective marginals for these tests.
        B, D = self._stds.shape
        cov = np.zeros((B, D, D), dtype=self._stds.dtype)
        for t in range(D):
            cov[:, t, t] = self._stds[:, t] ** 2
        return cov

    def sample(self, n_samples: int) -> torch.Tensor:
        B, D = self._mus_t.shape
        eps = torch.randn(n_samples, B, D)
        return self._mus_t[None, :, :] + eps * self._sigmas_t[None, :, :]


# ---------- ehvi_samples ----------

def test_ehvi_samples_dominated_returns_zero():
    """A sample strictly dominated by the front contributes nothing."""
    pf = np.array([[1.0, 1.0]], dtype=np.float32)
    ref = np.array([5.0, 5.0], dtype=np.float32)
    base_hv = (ref[0] - pf[0, 0]) * (ref[1] - pf[0, 1])

    dominated = np.array([3.0, 3.0], dtype=np.float32)
    assert ehvi_samples(dominated, base_hv, pf, ref) == 0.0


def test_ehvi_samples_beyond_ref_returns_zero():
    """A sample with any coord >= ref contributes nothing (HV undefined past ref)."""
    pf = np.array([[1.0, 1.0]], dtype=np.float32)
    ref = np.array([5.0, 5.0], dtype=np.float32)
    base_hv = (ref[0] - pf[0, 0]) * (ref[1] - pf[0, 1])

    s = np.array([6.0, 0.5], dtype=np.float32)  # past ref in obj1
    assert ehvi_samples(s, base_hv, pf, ref) == 0.0


def test_ehvi_samples_improving_sample_returns_positive():
    """A sample better than the front in both dims should give a positive contribution."""
    pf = np.array([[2.0, 2.0]], dtype=np.float32)
    ref = np.array([5.0, 5.0], dtype=np.float32)
    base_hv = (ref[0] - pf[0, 0]) * (ref[1] - pf[0, 1])  # 9.0

    s = np.array([1.0, 1.0], dtype=np.float32)
    out = ehvi_samples(s, base_hv, pf, ref)
    # New front is just (1,1), HV = (5-1)(5-1) = 16; improvement = 16 - 9 = 7
    assert out == pytest.approx(7.0, abs=1e-6)


def test_ehvi_samples_partial_improvement():
    """Sample non-dominated but not strictly better than every front member."""
    pf = np.array([[1.0, 3.0], [3.0, 1.0]], dtype=np.float32)
    ref = np.array([5.0, 5.0], dtype=np.float32)
    # base HV (in min space): rectangles for (1,3) and (3,1) up to ref
    # easier to compute via pygmo
    from pygmo import hypervolume
    base_hv = hypervolume(pf).compute(ref)

    s = np.array([2.0, 2.0], dtype=np.float32)  # adds a step in the staircase
    out = ehvi_samples(s, base_hv, pf, ref)
    assert out > 0
    # consistency: explicit HV of the extended front
    new_hv = hypervolume(np.vstack([pf, s])).compute(ref)
    assert out == pytest.approx(new_hv - base_hv, abs=1e-6)


# ---------- monte_carlo_ehvi_batch ----------

def test_mc_batch_returns_shape_B():
    torch.manual_seed(0)
    np.random.seed(0)

    pf = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]], dtype=np.float32)
    _, ref = front_augmentation(pf, front="lower", ref_mode="frac", frac=0.5, return_ref=True)
    from pygmo import hypervolume
    base_hv = hypervolume(pf).compute(ref)

    mus = np.array([[0.5, 0.5], [4.5, 4.5]], dtype=np.float32)
    sigmas = np.full_like(mus, 0.1)
    pool = _FakePoolPosterior(mus, sigmas)

    out = monte_carlo_ehvi_batch(
        pool, pf, ref, base_hv,
        front="lower", min_samples=64, max_samples=256, chunk_size=64,
    )
    assert out.shape == (2,)


def test_mc_batch_zero_for_certain_dominated_candidate():
    """A candidate whose mean is deep in the dominated region with tiny std should
    converge to ~0 EHVI."""
    torch.manual_seed(0)
    np.random.seed(0)

    pf = np.array([[0.0, 0.0]], dtype=np.float32)  # ideal-point front
    _, ref = front_augmentation(pf, front="lower", ref_mode="frac", frac=0.5, return_ref=True)
    base_hv = float((ref[0] - 0.0) * (ref[1] - 0.0))

    mus = np.array([[float(ref[0]) - 1e-3, float(ref[1]) - 1e-3]], dtype=np.float32)
    sigmas = np.array([[1e-3, 1e-3]], dtype=np.float32)
    pool = _FakePoolPosterior(mus, sigmas)

    out = monte_carlo_ehvi_batch(
        pool, pf, ref, base_hv,
        front="lower", min_samples=64, max_samples=256, chunk_size=64,
    )
    assert out[0] < 1e-2, f"expected ~0 for dominated candidate, got {out[0]}"


def test_mc_batch_positive_for_certain_improving_candidate():
    """Tiny-sigma candidate firmly inside the improvement region: positive EHVI."""
    torch.manual_seed(0)
    np.random.seed(0)

    pf = np.array([[2.0, 2.0]], dtype=np.float32)
    _, ref = front_augmentation(pf, front="lower", ref_mode="frac", frac=0.5, return_ref=True)
    base_hv = float((ref[0] - 2.0) * (ref[1] - 2.0))

    mus = np.array([[0.5, 0.5]], dtype=np.float32)
    sigmas = np.array([[1e-3, 1e-3]], dtype=np.float32)
    pool = _FakePoolPosterior(mus, sigmas)

    out = monte_carlo_ehvi_batch(
        pool, pf, ref, base_hv,
        front="lower", min_samples=64, max_samples=256, chunk_size=64,
    )
    assert out[0] > 0.0


def test_mc_batch_upper_lower_negation_symmetry():
    """
    The `front="upper"` branch negates posterior samples before the EHVI kernel.
    So feeding the model with negated means under "upper" must match feeding
    the same magnitude means under "lower" — both effective samples land in the
    same place in MIN space. This is the contract that lets a max-space GP feed
    into a min-space EHVI pipeline.
    """
    torch.manual_seed(0)
    np.random.seed(0)

    pf_min = np.array([[2.0, 2.0]], dtype=np.float32)
    _, ref = front_augmentation(pf_min, front="lower", ref_mode="frac", frac=0.5, return_ref=True)
    base_hv = float((ref[0] - 2.0) * (ref[1] - 2.0))

    mus_min = np.array([[0.5, 0.5]], dtype=np.float32)
    sigmas  = np.array([[1e-3, 1e-3]], dtype=np.float32)

    # min-space pool -> "lower"
    torch.manual_seed(123)
    e_lower = monte_carlo_ehvi_batch(
        _FakePoolPosterior(mus_min, sigmas), pf_min, ref, base_hv,
        front="lower", min_samples=128, max_samples=256, chunk_size=64,
    )[0]
    # max-space pool (sign-flipped means) -> "upper", internal negation undoes it
    torch.manual_seed(123)
    e_upper = monte_carlo_ehvi_batch(
        _FakePoolPosterior(-mus_min, sigmas), pf_min, ref, base_hv,
        front="upper", min_samples=128, max_samples=256, chunk_size=64,
    )[0]

    assert abs(e_lower - e_upper) < 1e-2, f"e_lower={e_lower}, e_upper={e_upper}"
