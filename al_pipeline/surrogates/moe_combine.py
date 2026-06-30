"""
MoE combination rules.

Three rules for collapsing (PS expert, nonPS expert) into a single prediction,
plus the law-of-total-variance moment match for the soft case.

These functions take numpy arrays of *matching shape* — typically (B,) for
per-candidate predictions or (B, 2) for both objectives at once. No magic
broadcasting beyond standard numpy rules.

Reference: this mirrors the MCSC `moe_utils.combine_*` semantics, ported here
without behavior changes so existing MoE checkpoints predict the same way.
"""
from __future__ import annotations

import numpy as np


def combine_soft(p_ps: np.ndarray, pred_ps: np.ndarray, pred_nonps: np.ndarray) -> np.ndarray:
    """
    Soft mixture: ``p * PS + (1 - p) * nonPS``.

    Returns the expected value under a 2-component Bernoulli(p) gate. This is
    the natural choice when the gate is calibrated and we want to average over
    its uncertainty rather than commit to one expert.
    """
    p = np.asarray(p_ps, dtype=np.float64)
    return p * np.asarray(pred_ps, dtype=np.float64) + (1.0 - p) * np.asarray(pred_nonps, dtype=np.float64)


def combine_hard(p_ps: np.ndarray, pred_ps: np.ndarray, pred_nonps: np.ndarray, threshold: float) -> np.ndarray:
    """
    Hard gate: PS expert where ``p_ps >= threshold``, else nonPS expert.

    Discrete switch — produces a piecewise constant prediction across the gate
    boundary. Useful for "discover PS-only" acquisition, where we commit to
    the PS expert in every candidate the gate confidently labels PS.
    """
    use_ps = np.asarray(p_ps, dtype=np.float64) >= float(threshold)
    return np.where(use_ps, np.asarray(pred_ps, dtype=np.float64), np.asarray(pred_nonps, dtype=np.float64))


def ps_guarded(p_ps: np.ndarray, pred_ps: np.ndarray, threshold: float) -> np.ndarray:
    """
    PS-guarded: PS expert where ``p_ps >= threshold``, else NaN (abstain).

    Not currently surfaced through the GA acquisition (the GA has no natural
    handler for NaN fitness), but kept for diagnostic / beam-search consumers
    that explicitly want a "PS-only" prediction column.
    """
    use_ps = np.asarray(p_ps, dtype=np.float64) >= float(threshold)
    return np.where(use_ps, np.asarray(pred_ps, dtype=np.float64), np.nan)


def soft_mixture_variance(
    p_ps: np.ndarray,
    mu_ps: np.ndarray,
    var_ps: np.ndarray,
    mu_nonps: np.ndarray,
    var_nonps: np.ndarray,
) -> np.ndarray:
    """
    Law-of-total-variance moment match for a 2-component Gaussian mixture.

    .. math::
        \\mathrm{Var}_{\\mathrm{mix}} =
        p (\\sigma_{\\mathrm{PS}}^2 + \\mu_{\\mathrm{PS}}^2)
        + (1-p) (\\sigma_{\\mathrm{nonPS}}^2 + \\mu_{\\mathrm{nonPS}}^2)
        - \\mu_{\\mathrm{mix}}^2

    Collapses the bimodal mixture to a single Gaussian by matching mean and
    variance. Analytic EHVI consumes this directly. NOT a calibrated
    uncertainty — for proper MC over the mixture, sample from each expert
    separately and gate per draw (see `MoEPoolPosterior.sample`).

    Only meaningful when both experts share one label scaler (i.e. predictions
    live in a common space). The MoE surrogate enforces ``label_scaler_scope='all'``.
    """
    p = np.asarray(p_ps, dtype=np.float64)
    mu_ps = np.asarray(mu_ps, dtype=np.float64)
    mu_nonps = np.asarray(mu_nonps, dtype=np.float64)
    var_ps = np.asarray(var_ps, dtype=np.float64)
    var_nonps = np.asarray(var_nonps, dtype=np.float64)
    mu_mix = p * mu_ps + (1.0 - p) * mu_nonps
    return p * (var_ps + mu_ps ** 2) + (1.0 - p) * (var_nonps + mu_nonps ** 2) - mu_mix ** 2
