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

  - `Surrogate.predict_design(X_raw) -> DesignPrediction`
    Beam-search-facing surface. Returns z-space + physical-space (via
    persisted per-expert label scalers) + gate (if MoE) + per-expert
    breakdown (if MoE). The beam ranks in quantile space derived from
    physical, so physical is a required output. See §III.6 of the
    beam-search reimplementation plan for the mixture-mechanics rationale.

  - `Surrogate.predict_design_sampled(X_raw, n_samples)` (optional)
    Sample from the z-distribution, inverse-transform each sample, then
    average in physical. Yields the unbiased physical mean E[Y] rather than
    the point-estimate s⁻¹(E[Z]) that `predict_design` returns. Only meant
    for a small validation-endpoint set — never called in the beam hot loop.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class DesignPrediction:
    """Prediction payload consumed by the beam search.

    All arrays are ``(B, 2)`` with column order ``(obj1, obj2)`` matching the
    surrogate's training configuration, unless noted. Physical values are in
    the raw label units (e.g. g/mL for ``exp_density``, model-specific units
    for ``diff``) — the beam runs a QuantileTransformer on ``phys_mean``
    before ranking.

    Attributes
    ----------
    z_mean, z_std
        Marginal per-objective mean and std in the surrogate's normalized
        (YJ + standardize) output space. ``z_std`` is the per-expert std for
        single-expert surrogates; for MoE it is the mixture marginal std
        matching ``sigma_z``.
    sigma_z
        Alias of ``z_std`` kept explicit to match the plan's III.6 notation.
        For MoE-soft this is the law-of-total-variance mixture σ (includes
        the between-expert term). For anchored / hard / global surrogates it
        equals the underlying expert's ``z_std``.
    phys_mean
        Inverse-scaled ``z_mean`` in physical units. This is the *point
        estimate* ``s⁻¹(E[Z])`` — under a nonlinear label transform (YJ) it
        is closer to the physical median than the mean. Use
        ``predict_design_sampled`` when an unbiased ``E[Y]`` is required for
        validation-endpoint comparison with simulation. May be ``None`` for
        surrogates that do not persist label scalers.
    phys_std
        Physical-space marginal std. ``None`` for the deterministic
        ``predict_design`` path (delta-method transformation from ``z_std``
        is deferred to reliability code that needs it). Populated by
        ``predict_design_sampled`` from the sample dispersion.
    p_ps
        Per-candidate gate probability of PS regime under the calibrated RF.
        ``None`` for surrogates without a gate (e.g. global GPR).
    per_expert
        Per-expert breakdown for MoE surrogates. Keys are ``"ps"`` and
        ``"nonps"`` (the bundle intentionally does not carry an ``"all"``
        expert — global-GPR comparisons go through a separately-loaded
        `GlobalGPRSurrogate`). Each value is a nested dict with ``z_mean``,
        ``z_std``, ``phys_mean``. ``None`` for non-MoE surrogates.
    """

    z_mean: np.ndarray
    z_std: np.ndarray
    sigma_z: np.ndarray
    phys_mean: np.ndarray | None
    phys_std: np.ndarray | None
    p_ps: np.ndarray | None
    per_expert: dict[str, dict[str, np.ndarray]] | None


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

    @abstractmethod
    def predict_design(self, X_raw: pd.DataFrame) -> DesignPrediction:
        """
        Beam-search-facing prediction surface.

        Same input contract as ``predict_pool`` — raw feature rows straight
        off the featurizer. Returns a `DesignPrediction` carrying z-space +
        physical + (optionally) gate + per-expert outputs. The beam uses the
        physical mean to feed a quantile transform for distance ranking; the
        gate + per-expert values feed the anchored / hard / expert-tied
        policies without reaching around the surrogate into the bundle.
        """

    def predict_design_sampled(
        self, X_raw: pd.DataFrame, *, n_samples: int = 200
    ) -> DesignPrediction:
        """
        Unbiased physical-space prediction via sampling.

        Draws ``n_samples`` from the surrogate's joint z-posterior,
        inverse-transforms each sample through the persisted per-expert
        label scalers, and averages in physical to yield ``E[Y]`` and its
        marginal std. Costs ``n_samples`` inverse-transform calls per
        batch — only meant for the validation-endpoint set (§III.6).

        Default implementation raises ``NotImplementedError``; surrogates
        that persist label scalers override.
        """
        del X_raw, n_samples  # subclasses that support sampling override this
        raise NotImplementedError(
            f"{type(self).__name__}.predict_design_sampled is not implemented"
        )
