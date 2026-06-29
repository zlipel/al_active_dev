"""
Parity test between the analytic stripe-decomposed EHVI and the Monte Carlo EHVI.

For the same posterior (means + diagonal covariance), the two should agree within
the MC standard error. This is the most important integration test for the
acquisition module — it pins down indexing, sign conventions, and the stripe
geometry against the integral definition of EHVI as E[HV improvement].
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from pygmo import hypervolume

from al_pipeline.acquisition.ehvi import (
    ehvi_analytic,
    monte_carlo_ehvi_batch,
    front_augmentation,
)


# Fixed-mean fixed-std posterior; same as the one used in test_ehvi_mc but kept
# local here to avoid a cross-test fixture import.
class _GaussianPosterior:
    def __init__(self, mus: torch.Tensor, sigmas: torch.Tensor):
        self.mus = mus
        self.sigmas = sigmas

    def rsample(self, sample_shape):
        S = int(sample_shape[0])
        B, D = self.mus.shape
        eps = torch.randn(S, B, D)
        return self.mus[None, :, :] + eps * self.sigmas[None, :, :]


class _FakeModel:
    def __init__(self, mus, sigmas):
        self._mus = torch.as_tensor(mus, dtype=torch.float32)
        self._sigmas = torch.as_tensor(sigmas, dtype=torch.float32)

    def __call__(self, _x):
        return _GaussianPosterior(self._mus, self._sigmas)


@pytest.mark.parametrize(
    "mu, sigma",
    [
        ((1.0, 1.0), (0.3, 0.3)),  # well inside improving region
        ((1.5, 0.5), (0.5, 0.5)),  # off-center
        ((2.5, 0.5), (0.3, 0.4)),  # near pareto front member
    ],
)
def test_analytic_mc_parity_single_candidate(mu, sigma):
    """For one candidate, MC EHVI converges to the analytic value."""
    torch.manual_seed(42)
    np.random.seed(42)

    pf_min = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]], dtype=np.float32)
    aug, ref = front_augmentation(pf_min, front="lower", ref_mode="frac", frac=0.5, return_ref=True)

    mu_arr = np.array([mu], dtype=np.float32)
    sig_arr = np.array([sigma], dtype=np.float32)

    e_analytic = float(ehvi_analytic(
        mu_arr[:, 0], sig_arr[:, 0], mu_arr[:, 1], sig_arr[:, 1], aug,
    )[0])

    base_hv = hypervolume(pf_min).compute(ref)
    model = _FakeModel(mu_arr, sig_arr)
    candidates = torch.zeros(1, 3)
    e_mc = float(monte_carlo_ehvi_batch(
        candidates, model, pf_min, ref, base_hv,
        front="lower",
        min_samples=512, max_samples=4096, stderr_tol=5e-3, chunk_size=256,
    )[0])

    # 3-sigma envelope around the analytic value; tolerance reflects MC stderr.
    assert abs(e_mc - e_analytic) < 5e-2, (
        f"MC and analytic disagree: mc={e_mc}, analytic={e_analytic}"
    )


def test_analytic_mc_parity_batch_ordering():
    """Across a small batch, both methods should rank candidates the same way."""
    torch.manual_seed(7)
    np.random.seed(7)

    pf_min = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]], dtype=np.float32)
    aug, ref = front_augmentation(pf_min, front="lower", ref_mode="frac", frac=0.5, return_ref=True)
    base_hv = hypervolume(pf_min).compute(ref)

    mus = np.array(
        [
            [1.0, 1.0],  # firmly improving
            [2.5, 2.5],  # ~ on the staircase
            [4.0, 4.0],  # past ref / dominated
        ],
        dtype=np.float32,
    )
    sigmas = np.full_like(mus, 0.3)

    e_analytic = ehvi_analytic(mus[:, 0], sigmas[:, 0], mus[:, 1], sigmas[:, 1], aug)

    model = _FakeModel(mus, sigmas)
    candidates = torch.zeros(mus.shape[0], 3)
    e_mc = monte_carlo_ehvi_batch(
        candidates, model, pf_min, ref, base_hv,
        front="lower",
        min_samples=512, max_samples=4096, stderr_tol=5e-3, chunk_size=256,
    )

    # The ranking (argsort, descending) must match.
    rank_a = np.argsort(-e_analytic)
    rank_m = np.argsort(-e_mc)
    assert (rank_a == rank_m).all(), f"analytic={e_analytic}, mc={e_mc}"
