"""Row 8 tests for the ``predict_design`` beam-facing prediction surface.

Structural + semantic pins:

- Shape: ``predict_design(X)`` returns arrays of ``(B, 2)`` (means / stds /
  physical) plus a ``(B,)`` gate for MoE.
- Physical inversion: ``phys_mean = s⁻¹(z_mean)`` for each expert via the
  persisted `label_scaler1` / `label_scaler2`. Under
  ``label_scaler_scope='all'`` both experts share scalers, so the blended
  ``phys_mean`` equals ``s⁻¹(soft-blended z_mean)`` computed once.
- Bias correction: ``predict_design_sampled`` yields ``E[Y]`` from
  z-samples inverse-transformed per sample. Under YJ, this differs from
  the ``predict_design`` point estimate ``s⁻¹(E[Z])`` — the sampled mean is
  the unbiased expectation the validation-endpoint metric should use.
- Global surrogate: ``predict_design`` returns z-only (``phys_mean=None``)
  because kfold_training does not persist scalers yet (deferred).

Fixtures come from ``test_moe._build_moe_bundle`` — trained on 24 synthetic
sequences with an MoE-consistent gate, sufficient to exercise every code
path in ``predict_design`` without pulling live checkpoints.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from al_pipeline.surrogates import (
    DesignPrediction,
    MoESurrogate,
)

# Reuse the synthetic-bundle scaffolding already in test_moe.
from tests.al_pipeline.test_moe import (
    _build_moe_bundle,
    _make_raw_features_df,
)


# ---------------------------------------------------------------------------
# MoE.predict_design — shape + per-expert breakdown
# ---------------------------------------------------------------------------

def test_predict_design_returns_expected_shapes_and_keys():
    bundle = _build_moe_bundle(seed=0)
    surr = MoESurrogate(bundle=bundle, policy="soft")
    X = _make_raw_features_df(6, seed=99)

    pred = surr.predict_design(X)

    assert isinstance(pred, DesignPrediction)
    B = len(X)
    assert pred.z_mean.shape == (B, 2)
    assert pred.z_std.shape == (B, 2)
    assert pred.sigma_z.shape == (B, 2)
    assert pred.phys_mean is not None and pred.phys_mean.shape == (B, 2)
    assert pred.phys_std is None  # deterministic path — no sampled std
    assert pred.p_ps is not None and pred.p_ps.shape == (B,)
    assert pred.per_expert is not None
    assert set(pred.per_expert.keys()) == {"ps", "nonps"}
    for regime in ("ps", "nonps"):
        pe = pred.per_expert[regime]
        assert set(pe.keys()) == {"z_mean", "z_std", "phys_mean"}
        assert pe["z_mean"].shape == (B, 2)
        assert pe["z_std"].shape == (B, 2)
        assert pe["phys_mean"].shape == (B, 2)


def test_predict_design_p_ps_and_stds_are_nonnegative():
    bundle = _build_moe_bundle(seed=1)
    surr = MoESurrogate(bundle=bundle, policy="soft")
    X = _make_raw_features_df(8, seed=200)

    pred = surr.predict_design(X)

    assert np.all(pred.p_ps >= 0.0) and np.all(pred.p_ps <= 1.0)
    assert np.all(pred.z_std >= 0.0)
    assert np.all(pred.sigma_z >= 0.0)
    for regime in ("ps", "nonps"):
        assert np.all(pred.per_expert[regime]["z_std"] >= 0.0)


def test_predict_design_soft_z_mean_matches_p_weighted_blend():
    """Blended z_mean == p·μ_PS + (1-p)·μ_nonPS to numerical precision."""
    bundle = _build_moe_bundle(seed=2)
    surr = MoESurrogate(bundle=bundle, policy="soft")
    X = _make_raw_features_df(10, seed=300)

    pred = surr.predict_design(X)
    p = pred.p_ps[:, None]
    expected = p * pred.per_expert["ps"]["z_mean"] + (1.0 - p) * pred.per_expert["nonps"]["z_mean"]
    np.testing.assert_allclose(pred.z_mean, expected, rtol=1e-12, atol=1e-12)


def test_predict_design_hard_z_mean_switches_at_threshold():
    """Under hard policy, per-candidate z_mean equals the winning expert's."""
    bundle = _build_moe_bundle(seed=3)
    surr = MoESurrogate(bundle=bundle, policy="hard", threshold=0.5)
    X = _make_raw_features_df(10, seed=400)

    pred = surr.predict_design(X)
    use_ps = (pred.p_ps >= 0.5)[:, None]
    expected = np.where(use_ps, pred.per_expert["ps"]["z_mean"], pred.per_expert["nonps"]["z_mean"])
    np.testing.assert_allclose(pred.z_mean, expected, rtol=1e-12, atol=1e-12)


def test_predict_design_phys_mean_inverts_z_mean_via_shared_scaler():
    """phys_mean == inverse-scale(z_mean) — the point-estimate path (III.6)."""
    bundle = _build_moe_bundle(seed=4)
    surr = MoESurrogate(bundle=bundle, policy="soft")
    X = _make_raw_features_df(6, seed=500)

    pred = surr.predict_design(X)
    # Manually apply the shared scaler (both experts hold identical instances
    # under label_scaler_scope='all').
    scaler1 = bundle.ps_expert.label_scaler1
    scaler2 = bundle.ps_expert.label_scaler2
    expected_phys = np.column_stack([
        scaler1.inverse_transform(pred.z_mean[:, [0]]).ravel(),
        scaler2.inverse_transform(pred.z_mean[:, [1]]).ravel(),
    ])
    np.testing.assert_allclose(pred.phys_mean, expected_phys, rtol=1e-12, atol=1e-12)


def test_predict_design_per_expert_phys_is_expert_only():
    """per_expert[r].phys_mean uses only regime r's expert's z_mean."""
    bundle = _build_moe_bundle(seed=5)
    surr = MoESurrogate(bundle=bundle, policy="soft")
    X = _make_raw_features_df(6, seed=600)

    pred = surr.predict_design(X)
    for regime in ("ps", "nonps"):
        exp = bundle.ps_expert if regime == "ps" else bundle.nonps_expert
        expected = exp.inverse_scale_z(pred.per_expert[regime]["z_mean"])
        np.testing.assert_allclose(
            pred.per_expert[regime]["phys_mean"], expected,
            rtol=1e-12, atol=1e-12,
        )


# ---------------------------------------------------------------------------
# MoE.predict_design_sampled — unbiased E[Y] via inverse-transform-then-mean
# ---------------------------------------------------------------------------

def test_predict_design_sampled_shapes():
    bundle = _build_moe_bundle(seed=6)
    surr = MoESurrogate(bundle=bundle, policy="soft")
    X = _make_raw_features_df(4, seed=700)

    pred = surr.predict_design_sampled(X, n_samples=50)

    B = len(X)
    assert pred.phys_mean is not None and pred.phys_mean.shape == (B, 2)
    assert pred.phys_std is not None and pred.phys_std.shape == (B, 2)
    # Sampled std is nonneg + strictly positive somewhere (posterior isn't collapsed)
    assert np.all(pred.phys_std >= 0.0)
    assert np.any(pred.phys_std > 0.0)


def test_predict_design_sampled_rejects_n_samples_too_small():
    bundle = _build_moe_bundle(seed=7)
    surr = MoESurrogate(bundle=bundle, policy="soft")
    X = _make_raw_features_df(2, seed=800)
    with pytest.raises(ValueError, match="n_samples"):
        surr.predict_design_sampled(X, n_samples=1)


def test_predict_design_sampled_e_y_differs_from_inverse_of_e_z():
    """Under YJ, s⁻¹(E[Z]) ≠ E[Y]. The sampled mean is the unbiased E[Y].

    High-n MC sample: the sampled ``phys_mean`` should differ from the
    deterministic ``predict_design.phys_mean`` (``s⁻¹(E[Z])``) by more than
    numerical precision on at least one candidate. This proves the sampling
    path is doing something non-trivial (not just re-emitting the point
    estimate).
    """
    bundle = _build_moe_bundle(seed=8, n_train=32)
    surr = MoESurrogate(bundle=bundle, policy="soft")
    X = _make_raw_features_df(6, seed=900)

    det = surr.predict_design(X)
    # Deterministic sampling for repeatability
    torch.manual_seed(0)
    samp = surr.predict_design_sampled(X, n_samples=1000)

    # z_mean etc. are inherited from `predict_design`, so they should match exactly.
    np.testing.assert_allclose(samp.z_mean, det.z_mean, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(samp.z_std, det.z_std, rtol=1e-12, atol=1e-12)

    # phys_mean should differ from det.phys_mean (Jensen shift under YJ).
    # We don't pin the magnitude — just that the two paths aren't identical.
    diff = np.abs(samp.phys_mean - det.phys_mean)
    assert np.any(diff > 1e-6), (
        "Sampled phys_mean equals s⁻¹(E[Z]) — sampling path likely bypassed"
    )


# ---------------------------------------------------------------------------
# GPRExpert.inverse_scale_z — round-trip
# ---------------------------------------------------------------------------

def test_gpr_expert_inverse_scale_z_round_trips_physical():
    """Feeding physical labels through the scaler, then back through
    ``inverse_scale_z``, recovers them within numerical precision."""
    bundle = _build_moe_bundle(seed=9)
    exp = bundle.ps_expert
    rng = np.random.default_rng(0)
    phys = np.column_stack([
        rng.uniform(-0.5, 0.5, size=8),
        rng.uniform(1.0, 2.0, size=8),
    ])
    z = np.column_stack([
        exp.label_scaler1.transform(phys[:, [0]]).ravel(),
        exp.label_scaler2.transform(phys[:, [1]]).ravel(),
    ])
    if exp.transform == "log":
        # inverse would need `log` applied to phys[:, 1] first — but our
        # synthetic bundle is 'yeoj' by default.
        pytest.skip("round-trip check only pinned for yeoj transform")
    round_trip = exp.inverse_scale_z(z)
    np.testing.assert_allclose(round_trip, phys, rtol=1e-9, atol=1e-9)


def test_gpr_expert_inverse_scale_z_rejects_wrong_shape():
    bundle = _build_moe_bundle(seed=10)
    exp = bundle.ps_expert
    with pytest.raises(ValueError, match=r"shape \(B, 2\)"):
        exp.inverse_scale_z(np.zeros((4, 3)))
    with pytest.raises(ValueError, match=r"shape \(B, 2\)"):
        exp.inverse_scale_z(np.zeros((4,)))


# ---------------------------------------------------------------------------
# predict_design — regime skip (expert_tied / anchored_reject optimization)
# ---------------------------------------------------------------------------


def test_predict_design_regime_ps_skips_nonps_expert():
    """When ``regime='ps'``, the PS expert's per_expert entry matches the
    full-computation baseline; the nonPS entry is a NaN sentinel."""
    bundle = _build_moe_bundle(seed=11)
    surr = MoESurrogate(bundle=bundle, policy="soft")
    X = _make_raw_features_df(5, seed=42)

    baseline = surr.predict_design(X)
    ps_only = surr.predict_design(X, regime="ps")

    # PS expert matches the baseline exactly.
    np.testing.assert_allclose(
        ps_only.per_expert["ps"]["z_mean"], baseline.per_expert["ps"]["z_mean"],
        rtol=1e-12, atol=1e-12,
    )
    np.testing.assert_allclose(
        ps_only.per_expert["ps"]["phys_mean"], baseline.per_expert["ps"]["phys_mean"],
        rtol=1e-12, atol=1e-12,
    )
    # nonPS is skipped → NaN sentinel.
    assert np.all(np.isnan(ps_only.per_expert["nonps"]["z_mean"]))
    assert np.all(np.isnan(ps_only.per_expert["nonps"]["z_std"]))
    assert np.all(np.isnan(ps_only.per_expert["nonps"]["phys_mean"]))
    # Top-level fields equal the PS expert channel.
    np.testing.assert_allclose(
        ps_only.z_mean, ps_only.per_expert["ps"]["z_mean"], rtol=1e-12, atol=1e-12,
    )
    np.testing.assert_allclose(
        ps_only.phys_mean, ps_only.per_expert["ps"]["phys_mean"],
        rtol=1e-12, atol=1e-12,
    )


def test_predict_design_regime_nonps_skips_ps_expert():
    bundle = _build_moe_bundle(seed=12)
    surr = MoESurrogate(bundle=bundle, policy="hard")
    X = _make_raw_features_df(7, seed=43)

    baseline = surr.predict_design(X)
    nonps_only = surr.predict_design(X, regime="nonps")

    np.testing.assert_allclose(
        nonps_only.per_expert["nonps"]["z_mean"],
        baseline.per_expert["nonps"]["z_mean"],
        rtol=1e-12, atol=1e-12,
    )
    assert np.all(np.isnan(nonps_only.per_expert["ps"]["z_mean"]))
    assert np.all(np.isnan(nonps_only.per_expert["ps"]["phys_mean"]))
    np.testing.assert_allclose(
        nonps_only.z_mean, nonps_only.per_expert["nonps"]["z_mean"],
        rtol=1e-12, atol=1e-12,
    )


def test_predict_design_regime_none_matches_baseline():
    """Passing ``regime=None`` (the default) returns the full both-experts
    result — no regression on soft / hard blending."""
    bundle = _build_moe_bundle(seed=13)
    surr = MoESurrogate(bundle=bundle, policy="soft")
    X = _make_raw_features_df(4, seed=44)

    pred_default = surr.predict_design(X)
    pred_none = surr.predict_design(X, regime=None)

    for key in ("z_mean", "z_std", "phys_mean", "p_ps"):
        np.testing.assert_allclose(
            getattr(pred_default, key), getattr(pred_none, key),
            rtol=1e-12, atol=1e-12,
        )


def test_predict_design_regime_gate_still_computed():
    """The RF gate is cheap and always computed regardless of ``regime`` —
    ``expert_tied`` records p_ps as a drift diagnostic."""
    bundle = _build_moe_bundle(seed=14)
    surr = MoESurrogate(bundle=bundle, policy="soft")
    X = _make_raw_features_df(3, seed=45)

    baseline_p_ps = surr.predict_design(X).p_ps
    for r in ("ps", "nonps"):
        p_ps_skipped = surr.predict_design(X, regime=r).p_ps
        np.testing.assert_allclose(p_ps_skipped, baseline_p_ps, rtol=1e-12, atol=1e-12)


def test_predict_design_regime_bad_value_raises():
    bundle = _build_moe_bundle(seed=15)
    surr = MoESurrogate(bundle=bundle, policy="soft")
    X = _make_raw_features_df(3, seed=46)
    with pytest.raises(ValueError, match="regime must be"):
        surr.predict_design(X, regime="junk")


def test_global_predict_design_accepts_regime_and_ignores_it():
    """Signature-compat check: `GlobalGPRSurrogate.predict_design` accepts
    ``regime`` for ABC compliance but has no per-regime experts to skip."""
    from al_pipeline.surrogates import make_surrogate
    from tests.al_pipeline.test_surrogate_gpr_global import (
        _fake_cfg,
        _toy_multitask,
    )
    bundle, stats, raw_test_df = _toy_multitask(seed=16)
    surr = make_surrogate(
        _fake_cfg("gpr_multitask"),
        model_bundle=bundle,
        normalization_stats=stats,
    )

    pred_default = surr.predict_design(raw_test_df)
    pred_ps      = surr.predict_design(raw_test_df, regime="ps")
    pred_nonps   = surr.predict_design(raw_test_df, regime="nonps")

    for other in (pred_ps, pred_nonps):
        np.testing.assert_allclose(other.z_mean, pred_default.z_mean, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(other.z_std,  pred_default.z_std,  rtol=1e-12, atol=1e-12)
