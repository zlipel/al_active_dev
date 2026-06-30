"""
Surrogate abstraction for acquisition consumers.

The GA's EHVI loop used to poke the GP directly: build the model, call
`model(X)`, pull `.mean` / `.stddev` for the analytic path, or hand the model
into `monte_carlo_ehvi_batch` and let it call `posterior.rsample(...)` for the
MC path. That works for one surrogate type. It breaks the moment we want a
mixture-of-experts surrogate that blends a global GP with PS / nonPS experts
via an RF gate — the blending logic would have to live inside `run_ga.py`,
which has no business knowing about experts.

This module defines the interface every surrogate (global GPR today, MoE
tomorrow) has to satisfy:

  - `Surrogate.predict_pool(Xn) -> PoolPosterior`
    One call, gets back a posterior object for a whole pool of candidates.

  - `PoolPosterior` exposes `.means`, `.stds`, and `.sample(n)`.
    The analytic EHVI path reads means/stds. The MC path calls `.sample(n)`
    repeatedly in its convergence loop. The posterior is computed *once* by
    `predict_pool` and reused across sample chunks — single-task GPs that
    can't joint-sample raise from `.sample(...)`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch


class PoolPosterior(ABC):
    """
    Posterior distribution over a pool of candidates, in *normalized objective*
    space, columns ordered as (obj1, obj2).

    Concrete subclasses cache whatever's needed to answer both the analytic
    consumer (means/stds) and the MC consumer (joint samples) without
    recomputing the underlying GP / mixture posterior.
    """

    @property
    @abstractmethod
    def means(self) -> np.ndarray:
        """Posterior mean per candidate per objective. Shape (B, 2)."""

    @property
    @abstractmethod
    def stds(self) -> np.ndarray:
        """Posterior std per candidate per objective. Shape (B, 2)."""

    @abstractmethod
    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Draw `n_samples` joint samples from the posterior.

        Returns
        -------
        torch.Tensor
            Shape (n_samples, B, 2). Must respect the joint covariance across
            the two objectives — independent per-objective samples are not
            acceptable for MC-EHVI. Surrogates that cannot joint-sample (e.g.
            single-task GPR) should raise NotImplementedError.
        """


class Surrogate(ABC):
    """
    A predictor over the 2-objective space used by AL acquisition.

    Implementations:
      - GlobalGPRSurrogate — wraps the existing multitask / singletask GPR.
      - (future) MoESurrogate — global + PS + nonPS experts, blended by RF gate.

    The interface is deliberately tiny: one call to `predict_pool` per
    candidate batch, which returns an object that supports both analytic and
    MC consumers. Anything more would be premature.
    """

    @abstractmethod
    def predict_pool(self, Xn: np.ndarray) -> PoolPosterior:
        """
        Build a posterior over a batch of *already-normalized* feature vectors.

        Parameters
        ----------
        Xn : np.ndarray
            Shape (B, n_features). Features must already be normalized with
            whatever stats the surrogate expects (global stats for
            GlobalGPRSurrogate; per-expert stats for MoE in the future).

        Returns
        -------
        PoolPosterior
            Posterior over (obj1, obj2) for the B candidates.
        """

    @property
    @abstractmethod
    def supports_joint_sampling(self) -> bool:
        """
        Whether `predict_pool(...).sample(n)` will work.

        True for multitask GPR and MoE; False for the single-task GPR mode
        (which has no cross-objective covariance and is analytic-only).
        """
