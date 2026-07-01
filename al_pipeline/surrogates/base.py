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
import pandas as pd
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

    @property
    @abstractmethod
    def covariance(self) -> np.ndarray:
        """
        Per-candidate joint (2, 2) covariance matrix, shape (B, 2, 2).

        Consumed by the pessimism penalty in the augmentation path, which
        compares candidate uncertainty ellipses against previously selected
        children. Subclasses that have no cross-objective structure (e.g. the
        single-task GPR wrapper) return a diagonal fallback so consumers can
        still call the same code path — the pessimism math degrades to the
        marginal-variance case.
        """

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
      - MoESurrogate — global + PS + nonPS experts blended by an RF gate.

    The interface takes RAW features (as produced by `SequenceFeaturizer
    .featurize_many`). Each surrogate handles its own preprocessing internally:
    GlobalGPRSurrogate stores the global normalization stats and applies them;
    MoESurrogate routes through per-expert normalizers. The GA stays out of
    surrogate-specific preprocessing, which keeps the abstraction honest when
    a new surrogate (DNN, MoE variant, etc.) needs a different pipeline.
    """

    @abstractmethod
    def predict_pool(self, X_raw: pd.DataFrame) -> PoolPosterior:
        """
        Build a posterior over a batch of raw feature rows.

        Parameters
        ----------
        X_raw : pd.DataFrame
            Shape (B, n_features). Columns must be the standard featurizer
            output (the 20 AA columns + the engineered columns produced by
            `SequenceFeaturizer.featurize_many`). Each surrogate normalizes
            internally — the GA never has to know whether the surrogate uses
            global stats, per-expert stats, or no normalization at all.

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
