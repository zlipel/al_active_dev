"""Row 9 tests for `beam_search.policy.BeamPolicy`.

The tests build a small synthetic MoE bundle (reusing test_moe.py's
_build_moe_bundle scaffold) plus a QuantileTransformer over the synthetic
labels, then exercise every policy kind through `predict_candidates`.

Coverage:
  * Construction validation (kind, start_regime, surrogate type)
  * expert_tied / anchored_reject use the correct per-expert channels
  * anchored_reject flags gate-opposed candidates as ``rejected_by_gate``
  * soft blend matches `predict_design.z_mean` per-candidate
  * hard switch matches `predict_design.per_expert` at the given threshold
  * reason ledger populates as ``ok`` / ``invalid_phys`` / ``rejected_by_gate``
  * `predict_candidate_frames` adapter preserves the old dict shape and
    carries `p_ps` + `reason` forward for `beam_search_paths`.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import QuantileTransformer

from al_pipeline.featurization.sequence_featurizer import SequenceFeaturizer
from al_pipeline.surrogates import MoESurrogate

from beam_search.cross_paths.beam_search import predict_candidate_frames
from beam_search.policy import BeamPolicy, PolicyPrediction

from tests.al_pipeline.test_moe import (
    _build_moe_bundle,
    _make_raw_features_df,
    _make_synthetic_labels,
)


# ---------------------------------------------------------------------------
# Stand-in featurizer for BeamPolicy — takes list[str], returns raw DataFrame
# ---------------------------------------------------------------------------

class _DirectFeaturizer:
    """Featurizer that just returns a pre-computed DataFrame.

    Real ``SequenceFeaturizer`` (numba or serial) needs an FF database and
    real amino-acid sequences. Row 9 policy tests exercise scoring logic on
    synthetic feature rows without valid AA sequences, so this stand-in
    accepts index-into-precomputed-df signals instead of sequences and
    returns the matching feature rows verbatim.
    """
    def __init__(self, features_df):
        self._df = features_df.reset_index(drop=True)

    def featurize_many_fast(self, seqs, feat_threads, as_df):
        # Interpret each "sequence" as an integer row index into self._df.
        idx = [int(s) for s in seqs]
        out = self._df.iloc[idx].reset_index(drop=True)
        return out if as_df else out.to_numpy()


def _fit_quantile(labels_df):
    q_rho = QuantileTransformer(
        n_quantiles=min(1000, len(labels_df)),
        random_state=0,
        output_distribution="uniform",
    ).fit(labels_df[["exp_density"]].to_numpy())
    q_diff = QuantileTransformer(
        n_quantiles=min(1000, len(labels_df)),
        random_state=0,
        output_distribution="uniform",
    ).fit(labels_df[["diff"]].to_numpy())
    return q_rho, q_diff


def _pool_of_size(n, seed):
    """Build (features_df, labels_df, direct_featurizer, seqs_as_indices)."""
    feats_df = _make_raw_features_df(n, seed=seed)
    labels_df = _make_synthetic_labels(feats_df, seed=seed)
    fzr = _DirectFeaturizer(feats_df)
    seqs = [str(i) for i in range(n)]
    return feats_df, labels_df, fzr, seqs


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------

def test_beam_policy_rejects_unknown_kind():
    bundle = _build_moe_bundle(seed=0)
    surr = MoESurrogate(bundle=bundle, policy="soft")
    feats_df, labels_df, fzr, _ = _pool_of_size(6, seed=0)
    q_rho, q_diff = _fit_quantile(labels_df)
    with pytest.raises(ValueError, match="unknown BeamPolicy kind"):
        BeamPolicy(
            kind="totally_unknown",  # type: ignore[arg-type]
            surrogate=surr, featurizer=fzr, q_rho=q_rho, q_diff=q_diff,
        )


def test_beam_policy_expert_tied_requires_start_regime():
    bundle = _build_moe_bundle(seed=0)
    surr = MoESurrogate(bundle=bundle, policy="soft")
    feats_df, labels_df, fzr, _ = _pool_of_size(6, seed=0)
    q_rho, q_diff = _fit_quantile(labels_df)
    with pytest.raises(ValueError, match="requires start_regime"):
        BeamPolicy(
            kind="expert_tied",
            surrogate=surr, featurizer=fzr, q_rho=q_rho, q_diff=q_diff,
        )
    # anchored_reject also requires it
    with pytest.raises(ValueError, match="requires start_regime"):
        BeamPolicy(
            kind="anchored_reject",
            surrogate=surr, featurizer=fzr, q_rho=q_rho, q_diff=q_diff,
        )


def test_beam_policy_soft_hard_no_start_regime_required():
    bundle = _build_moe_bundle(seed=1)
    surr = MoESurrogate(bundle=bundle, policy="soft")
    feats_df, labels_df, fzr, seqs = _pool_of_size(6, seed=1)
    q_rho, q_diff = _fit_quantile(labels_df)
    # Should construct without start_regime and dispatch cleanly.
    for kind in ("soft", "hard"):
        BeamPolicy(
            kind=kind,
            surrogate=surr, featurizer=fzr, q_rho=q_rho, q_diff=q_diff,
        )


# ---------------------------------------------------------------------------
# predict_candidates shape + reason ledger
# ---------------------------------------------------------------------------

def test_predict_candidates_returns_expected_shapes_and_reasons():
    bundle = _build_moe_bundle(seed=2)
    surr = MoESurrogate(bundle=bundle, policy="soft")
    feats_df, labels_df, fzr, seqs = _pool_of_size(8, seed=2)
    q_rho, q_diff = _fit_quantile(labels_df)

    p = BeamPolicy(
        kind="expert_tied", start_regime="ps",
        surrogate=surr, featurizer=fzr, q_rho=q_rho, q_diff=q_diff,
    )
    out = p.predict_candidates(seqs)

    assert isinstance(out, PolicyPrediction)
    B = len(seqs)
    assert out.z_mean.shape == (B, 2)
    assert out.z_std.shape == (B, 2)
    assert out.phys.shape == (B, 2)
    assert out.uv.shape == (B, 2)
    assert out.valid.shape == (B,)
    assert out.reason.shape == (B,)
    assert set(np.unique(out.reason)).issubset({"ok", "invalid_phys", "rejected_by_gate"})
    # p_ps present for MoE surrogates
    assert out.p_ps is not None
    assert out.p_ps.shape == (B,)


def test_predict_candidates_empty_batch_returns_empty_arrays():
    bundle = _build_moe_bundle(seed=3)
    surr = MoESurrogate(bundle=bundle, policy="soft")
    feats_df, labels_df, fzr, _ = _pool_of_size(4, seed=3)
    q_rho, q_diff = _fit_quantile(labels_df)
    p = BeamPolicy(
        kind="soft",
        surrogate=surr, featurizer=fzr, q_rho=q_rho, q_diff=q_diff,
    )
    out = p.predict_candidates([])
    assert out.z_mean.shape == (0, 2)
    assert out.valid.shape == (0,)


# ---------------------------------------------------------------------------
# Kind-specific channel extraction
# ---------------------------------------------------------------------------

def test_expert_tied_ps_uses_ps_expert_channel():
    bundle = _build_moe_bundle(seed=4)
    surr = MoESurrogate(bundle=bundle, policy="soft")
    feats_df, labels_df, fzr, seqs = _pool_of_size(6, seed=4)
    q_rho, q_diff = _fit_quantile(labels_df)

    ref = surr.predict_design(feats_df.reset_index(drop=True))
    p = BeamPolicy(
        kind="expert_tied", start_regime="ps",
        surrogate=surr, featurizer=fzr, q_rho=q_rho, q_diff=q_diff,
    )
    out = p.predict_candidates(seqs)
    np.testing.assert_allclose(out.z_mean, ref.per_expert["ps"]["z_mean"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(out.phys, ref.per_expert["ps"]["phys_mean"], rtol=1e-12, atol=1e-12)


def test_expert_tied_nonps_uses_nonps_expert_channel():
    bundle = _build_moe_bundle(seed=5)
    surr = MoESurrogate(bundle=bundle, policy="soft")
    feats_df, labels_df, fzr, seqs = _pool_of_size(6, seed=5)
    q_rho, q_diff = _fit_quantile(labels_df)

    ref = surr.predict_design(feats_df.reset_index(drop=True))
    p = BeamPolicy(
        kind="expert_tied", start_regime="nonps",
        surrogate=surr, featurizer=fzr, q_rho=q_rho, q_diff=q_diff,
    )
    out = p.predict_candidates(seqs)
    np.testing.assert_allclose(out.z_mean, ref.per_expert["nonps"]["z_mean"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(out.phys, ref.per_expert["nonps"]["phys_mean"], rtol=1e-12, atol=1e-12)


def test_soft_matches_predict_design_blended():
    bundle = _build_moe_bundle(seed=6)
    surr = MoESurrogate(bundle=bundle, policy="soft")
    feats_df, labels_df, fzr, seqs = _pool_of_size(6, seed=6)
    q_rho, q_diff = _fit_quantile(labels_df)

    ref = surr.predict_design(feats_df.reset_index(drop=True))
    p = BeamPolicy(
        kind="soft",
        surrogate=surr, featurizer=fzr, q_rho=q_rho, q_diff=q_diff,
    )
    out = p.predict_candidates(seqs)
    np.testing.assert_allclose(out.z_mean, ref.z_mean, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(out.phys, ref.phys_mean, rtol=1e-12, atol=1e-12)


def test_hard_switches_per_candidate_at_threshold():
    bundle = _build_moe_bundle(seed=7)
    # Surrogate stays 'soft' — beam policy's threshold is independent of it.
    surr = MoESurrogate(bundle=bundle, policy="soft")
    feats_df, labels_df, fzr, seqs = _pool_of_size(8, seed=7)
    q_rho, q_diff = _fit_quantile(labels_df)

    ref = surr.predict_design(feats_df.reset_index(drop=True))
    p = BeamPolicy(
        kind="hard", hard_threshold=0.5,
        surrogate=surr, featurizer=fzr, q_rho=q_rho, q_diff=q_diff,
    )
    out = p.predict_candidates(seqs)
    use_ps = (ref.p_ps >= 0.5)[:, None]
    expected_z = np.where(use_ps, ref.per_expert["ps"]["z_mean"], ref.per_expert["nonps"]["z_mean"])
    expected_phys = np.where(use_ps, ref.per_expert["ps"]["phys_mean"], ref.per_expert["nonps"]["phys_mean"])
    np.testing.assert_allclose(out.z_mean, expected_z, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(out.phys, expected_phys, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# anchored_reject gate filter
# ---------------------------------------------------------------------------

def test_anchored_reject_rejects_confidently_opposite_gate():
    bundle = _build_moe_bundle(seed=8, n_train=32)
    surr = MoESurrogate(bundle=bundle, policy="soft")
    feats_df, labels_df, fzr, seqs = _pool_of_size(24, seed=8)
    q_rho, q_diff = _fit_quantile(labels_df)

    # PS-anchored: candidates with p_ps < 0.5 should be rejected.
    ref = surr.predict_design(feats_df.reset_index(drop=True))
    if np.all(ref.p_ps >= 0.5) or np.all(ref.p_ps < 0.5):
        pytest.skip("synthetic gate produced degenerate p_ps for this seed")

    p_ps_anchored = BeamPolicy(
        kind="anchored_reject", start_regime="ps", reject_threshold=0.5,
        surrogate=surr, featurizer=fzr, q_rho=q_rho, q_diff=q_diff,
    )
    out = p_ps_anchored.predict_candidates(seqs)

    should_reject = ref.p_ps < 0.5
    was_rejected = out.reason == "rejected_by_gate"
    # Every should_reject candidate that also passed physical validity
    # ends up as rejected_by_gate; physically-invalid ones stay invalid_phys.
    finite = np.isfinite(ref.per_expert["ps"]["phys_mean"][:, 0]) & (
        ref.per_expert["ps"]["phys_mean"][:, 0] > 1e-12
    ) & np.isfinite(ref.per_expert["ps"]["phys_mean"][:, 1]) & (
        ref.per_expert["ps"]["phys_mean"][:, 1] > 1e-12
    )
    expected_rejected = should_reject & finite
    np.testing.assert_array_equal(was_rejected, expected_rejected)
    # All rejected candidates have valid=False
    assert not np.any(out.valid[was_rejected])


def test_anchored_reject_nonps_start_reverses_polarity():
    """nonPS start rejects candidates with p_ps >= threshold (they look PS)."""
    bundle = _build_moe_bundle(seed=9, n_train=32)
    surr = MoESurrogate(bundle=bundle, policy="soft")
    feats_df, labels_df, fzr, seqs = _pool_of_size(24, seed=9)
    q_rho, q_diff = _fit_quantile(labels_df)

    ref = surr.predict_design(feats_df.reset_index(drop=True))
    if np.all(ref.p_ps >= 0.5) or np.all(ref.p_ps < 0.5):
        pytest.skip("synthetic gate produced degenerate p_ps for this seed")

    p = BeamPolicy(
        kind="anchored_reject", start_regime="nonps", reject_threshold=0.5,
        surrogate=surr, featurizer=fzr, q_rho=q_rho, q_diff=q_diff,
    )
    out = p.predict_candidates(seqs)

    # Candidates flagged PS by the gate get rejected under a nonPS anchor.
    should_reject = ref.p_ps >= 0.5
    # Filter to physically-valid ones (nonps expert side)
    finite = np.isfinite(ref.per_expert["nonps"]["phys_mean"][:, 0]) & (
        ref.per_expert["nonps"]["phys_mean"][:, 0] > 1e-12
    ) & np.isfinite(ref.per_expert["nonps"]["phys_mean"][:, 1]) & (
        ref.per_expert["nonps"]["phys_mean"][:, 1] > 1e-12
    )
    expected_rejected = should_reject & finite
    was_rejected = out.reason == "rejected_by_gate"
    np.testing.assert_array_equal(was_rejected, expected_rejected)


# ---------------------------------------------------------------------------
# label_scalers convenience
# ---------------------------------------------------------------------------

def test_label_scalers_available_for_moe_policies():
    bundle = _build_moe_bundle(seed=10)
    surr = MoESurrogate(bundle=bundle, policy="soft")
    feats_df, labels_df, fzr, _ = _pool_of_size(4, seed=10)
    q_rho, q_diff = _fit_quantile(labels_df)
    p = BeamPolicy(
        kind="expert_tied", start_regime="ps",
        surrogate=surr, featurizer=fzr, q_rho=q_rho, q_diff=q_diff,
    )
    s1, s2 = p.label_scalers
    # Same instances the bundle carries.
    assert s1 is bundle.ps_expert.label_scaler1
    assert s2 is bundle.ps_expert.label_scaler2


# ---------------------------------------------------------------------------
# predict_candidate_frames adapter
# ---------------------------------------------------------------------------

def test_predict_candidate_frames_preserves_old_dict_shape_plus_new_keys():
    bundle = _build_moe_bundle(seed=11)
    surr = MoESurrogate(bundle=bundle, policy="soft")
    feats_df, labels_df, fzr, seqs = _pool_of_size(6, seed=11)
    q_rho, q_diff = _fit_quantile(labels_df)
    p = BeamPolicy(
        kind="expert_tied", start_regime="ps",
        surrogate=surr, featurizer=fzr, q_rho=q_rho, q_diff=q_diff,
    )
    d = predict_candidate_frames(p, seqs)
    assert set(d.keys()) >= {"z", "phys", "uv", "valid", "p_ps", "reason"}
    assert d["z"].shape == (len(seqs), 2)
    assert d["phys"].shape == (len(seqs), 2)
    assert d["uv"].shape == (len(seqs), 2)
    assert d["valid"].shape == (len(seqs),)
    assert d["reason"].shape == (len(seqs),)
