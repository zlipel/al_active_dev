"""
Self-contained tests for the MoE surrogate (al_pipeline.surrogates.moe + gpr_expert
+ moe_combine).

No imports from MODEL_COMPARISON_STELLAR_CURR — every fixture is built inline
from synthetic data. The tests cover four levels:

1. `moe_combine` — pure-numpy combination rules and the soft mixture variance
   identity (matches a Monte Carlo estimate of Var[p*X + (1-p)*Y] when p is a
   Bernoulli random selector).

2. `GPRExpert` — train on raw rows, predict, save+load checkpoint, verify
   that the reloaded expert matches the original on the same inputs.

3. `MoEBundle` — three experts + RF gate, metadata validation refuses
   mismatched transform / scope / iter; `from_components` accepts a consistent
   bundle.

4. `MoESurrogate` — assembles a tiny end-to-end MoE on synthetic data, runs
   `predict_pool(raw_df)`, and pins:
     - shape contracts (means/stds (B,2); sample (n,B,2))
     - soft policy uses gate-weighted blends; hard policy picks one expert
     - sample() under soft policy reproduces the gate-weighted mean in
       expectation (high-n MC check)
     - scope='regime' is rejected at construction time
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import gpytorch
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PowerTransformer

from al_pipeline.surrogates import (
    GPRExpert,
    MoEBundle,
    MoEPoolPosterior,
    MoESurrogate,
    build_rf_features,
    classifier_p_ps,
    combine_hard,
    combine_soft,
    ps_guarded,
    soft_mixture_variance,
)


AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
FEATURE_COLUMNS = AMINO_ACIDS + [
    "length", "SCD", "SHD", "|net charge|", "sum lambda",
    "beads(+)", "beads(-)", "shan ent", "mol wt",
]
LABEL_COLUMNS = ["exp_density", "diff"]


# ---------- shared fixtures ----------

def _make_raw_features_df(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        L = int(rng.integers(20, 160 + 1))
        probs = rng.dirichlet(np.ones(20))
        counts = rng.multinomial(L, probs).astype(float)
        scd = float(rng.normal(0.0, 0.5))
        shd = float(rng.uniform(0.0, 2.5))
        net = float(rng.integers(0, 10))
        sum_lambda = float(rng.uniform(0.0, 3.5))
        beads_pos = float(rng.integers(0, max(1, L // 5)))
        beads_neg = float(rng.integers(0, max(1, L // 5)))
        shan_ent = float(rng.uniform(2.0, 4.5))
        mol_wt = float(L * 110.0 + rng.normal(0.0, 50.0))
        rows.append(list(counts) + [L, scd, shd, net, sum_lambda, beads_pos, beads_neg, shan_ent, mol_wt])
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def _make_synthetic_labels(features_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Labels that vary with features — gives the GP something to learn."""
    rng = np.random.default_rng(seed)
    n = len(features_df)
    # density correlates with SCD; diff is positive scalar (so log transform works)
    density = features_df["SCD"].to_numpy() + rng.normal(0.0, 0.1, n)
    diff = 1.0 + np.abs(features_df["sum lambda"].to_numpy()) + rng.uniform(0.0, 0.5, n)
    return pd.DataFrame({"exp_density": density, "diff": diff})


def _fit_scalers(labels_df: pd.DataFrame, transform: str):
    """Fit one scaler per objective, mirroring the global-scope MoE training prep."""
    y0 = labels_df[["exp_density"]].to_numpy(dtype=np.float64)
    y1 = labels_df[["diff"]].to_numpy(dtype=np.float64)
    if transform == "log":
        y1 = np.log(y1 + 1e-8)
    scaler1 = PowerTransformer(method="yeo-johnson", standardize=True)
    scaler2 = PowerTransformer(method="yeo-johnson", standardize=True)
    scaler1.fit(y0)
    scaler2.fit(y1)
    return scaler1, scaler2


def _train_tiny_expert(
    raw_features_df: pd.DataFrame,
    raw_labels_df: pd.DataFrame,
    scaler1, scaler2, transform: str = "yeoj",
    epochs: int = 30, seed: int = 0,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return GPRExpert.train(
        features_raw_df=raw_features_df,
        labels_raw_df=raw_labels_df,
        label_columns=LABEL_COLUMNS,
        transform=transform,
        scaler1=scaler1, scaler2=scaler2,
        feature_columns=FEATURE_COLUMNS,
        lr=0.1, epochs=epochs, patience=5,
    )


def _stamp_provenance(expert: GPRExpert, regime: str, scope: str = "all",
                       model_name: str = "TEST", iteration: int = 0) -> GPRExpert:
    """Fill in provenance fields that the bundle validator checks."""
    expert.label_scaler_scope = scope
    expert.model_name = model_name
    expert.iteration = iteration
    expert.regime = regime
    return expert


def _make_rf_bundle(features_df: pd.DataFrame, ps_labels: np.ndarray, seed: int = 0):
    """Train a tiny RF on raw->converted features. Returns the dict shape MoEBundle expects."""
    X_rf, conv_cols = build_rf_features(features_df, FEATURE_COLUMNS, None)
    rf = RandomForestClassifier(n_estimators=10, random_state=seed)
    rf.fit(X_rf, ps_labels)
    return {
        "classifier": rf,
        "rf_raw_feature_columns": FEATURE_COLUMNS,
        "rf_converted_feature_columns": conv_cols,
        "rf_feature_space": "converted_unstandardized",
        "ps_definition": "density > 0",
        "random_state": seed,
        "threshold": 0.5,
        "model_name": "TEST",
        "iter": 0,
        "transform": "yeoj",
        "label_scaler_scope": "all",
    }


def _build_moe_bundle(seed: int = 0, scope: str = "all", n_train: int = 24) -> MoEBundle:
    """End-to-end tiny MoE: train PS + nonPS experts, train an RF, return the validated bundle."""
    feats_df = _make_raw_features_df(n_train, seed)
    labels_df = _make_synthetic_labels(feats_df, seed)

    # PS rows: positive exp_density. Half-and-half by construction (labels are centered).
    is_ps = (labels_df["exp_density"] > 0).to_numpy().astype(int)

    scaler1, scaler2 = _fit_scalers(labels_df, transform="yeoj")

    ps_ex = _train_tiny_expert(feats_df[is_ps == 1].reset_index(drop=True),
                                labels_df[is_ps == 1].reset_index(drop=True), scaler1, scaler2)
    nps_ex = _train_tiny_expert(feats_df[is_ps == 0].reset_index(drop=True),
                                 labels_df[is_ps == 0].reset_index(drop=True), scaler1, scaler2)

    _stamp_provenance(ps_ex, "ps", scope=scope)
    _stamp_provenance(nps_ex, "nonps", scope=scope)

    rf_bundle = _make_rf_bundle(feats_df, is_ps, seed=seed)
    rf_bundle["label_scaler_scope"] = scope
    return MoEBundle.from_components(rf_bundle, ps_ex, nps_ex)


# ---------- moe_combine ----------

def test_combine_soft_blends_elementwise():
    p = np.array([0.2, 0.8])
    ps = np.array([10.0, 10.0])
    nps = np.array([0.0, 0.0])
    out = combine_soft(p, ps, nps)
    np.testing.assert_allclose(out, [2.0, 8.0])


def test_combine_hard_picks_per_threshold():
    p = np.array([0.1, 0.6, 0.95])
    ps = np.array([1.0, 1.0, 1.0])
    nps = np.array([-1.0, -1.0, -1.0])
    np.testing.assert_array_equal(combine_hard(p, ps, nps, threshold=0.5), [-1.0, 1.0, 1.0])


def test_ps_guarded_returns_nan_where_gate_fails():
    p = np.array([0.1, 0.6])
    ps = np.array([3.0, 4.0])
    out = ps_guarded(p, ps, threshold=0.5)
    assert np.isnan(out[0])
    assert out[1] == 4.0


def test_soft_mixture_variance_matches_mc_estimate():
    """
    Var[Z] where Z = X if Bernoulli(p) else Y, X~N(mu_ps,var_ps), Y~N(mu_nps,var_nps)
    must match the closed-form law-of-total-variance moment match. High-n MC.
    """
    rng = np.random.default_rng(0)
    p, mu_ps, var_ps, mu_nps, var_nps = 0.3, 2.0, 0.5, -1.0, 1.5
    n = 200_000
    z = rng.binomial(1, p, n).astype(bool)
    x = rng.normal(mu_ps, np.sqrt(var_ps), n)
    y = rng.normal(mu_nps, np.sqrt(var_nps), n)
    samples = np.where(z, x, y)
    var_mc = samples.var(ddof=0)
    var_closed = float(soft_mixture_variance(p, mu_ps, var_ps, mu_nps, var_nps))
    # 1% relative tolerance on a 200k MC estimate is generous.
    assert abs(var_mc - var_closed) / var_closed < 0.01


def test_soft_mixture_variance_is_nonnegative_at_boundaries():
    """p=0 -> Var_nps. p=1 -> Var_ps. No discontinuity at the endpoints."""
    for p, expected in ((0.0, 1.5), (1.0, 0.5)):
        out = float(soft_mixture_variance(p, 2.0, 0.5, -1.0, 1.5))
        assert out == pytest.approx(expected, abs=1e-9)


# ---------- RF feature pipeline ----------

def test_build_rf_features_returns_converted_unstandardized():
    """AA columns should be fractions (sum to 1 per row) — NOT standardized."""
    feats_df = _make_raw_features_df(10, seed=0)
    X_rf, conv_cols = build_rf_features(feats_df, FEATURE_COLUMNS, None)
    # The first 20 columns are AAs; their per-row sum is 1.0 in converted space.
    aa_sums = X_rf[:, :20].sum(axis=1)
    np.testing.assert_allclose(aa_sums, np.ones(10), rtol=1e-5, atol=1e-6)


def test_build_rf_features_rejects_missing_raw_columns():
    feats_df = _make_raw_features_df(5, seed=0).drop(columns=["length"])
    with pytest.raises(ValueError, match="missing raw columns"):
        build_rf_features(feats_df, FEATURE_COLUMNS, None)


def test_classifier_p_ps_handles_single_class_fold():
    """A degenerate fold where only the nonPS class was seen: P(PS|x) must be 0, not crash."""
    feats_df = _make_raw_features_df(20, seed=0)
    X_rf, _ = build_rf_features(feats_df, FEATURE_COLUMNS, None)
    rf = RandomForestClassifier(n_estimators=5, random_state=0)
    rf.fit(X_rf, np.zeros(20, dtype=int))   # all nonPS
    p = classifier_p_ps(rf, X_rf)
    np.testing.assert_array_equal(p, np.zeros(20))


# ---------- GPRExpert ----------

def test_gpr_expert_train_predict_shape_and_dtype():
    """One trained expert: predict returns the documented z-space dict with finite numbers."""
    feats = _make_raw_features_df(16, seed=1)
    labels = _make_synthetic_labels(feats, seed=1)
    scaler1, scaler2 = _fit_scalers(labels, "yeoj")
    expert = _train_tiny_expert(feats, labels, scaler1, scaler2)

    test_feats = _make_raw_features_df(7, seed=11)
    out = expert.predict(test_feats)

    for k in ("exp_density_z_mean", "exp_density_z_var", "exp_density_std_norm",
              "diff_z_mean", "diff_z_var", "diff_std_norm"):
        assert k in out
        assert out[k].shape == (7,)
        assert np.isfinite(out[k]).all()
    # predict() is z-space only — no phys keys.
    assert "exp_density_phys" not in out
    assert "diff_phys" not in out


def test_gpr_expert_checkpoint_roundtrip():
    """Save then load — reloaded expert must match the original on the same inputs."""
    feats = _make_raw_features_df(20, seed=2)
    labels = _make_synthetic_labels(feats, seed=2)
    scaler1, scaler2 = _fit_scalers(labels, "yeoj")
    expert = _train_tiny_expert(feats, labels, scaler1, scaler2)
    _stamp_provenance(expert, "all", scope="all")

    test_feats = _make_raw_features_df(5, seed=22)
    before = expert.predict(test_feats)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ckpt_path = td / "expert.pt"
        feats_path = td / "feats.csv"
        labels_path = td / "labels.csv"
        feats.to_csv(feats_path, index=False)
        labels.to_csv(labels_path, index=False)

        expert.save_checkpoint(
            str(ckpt_path),
            regime="all", label_scaler_scope="all",
            original_indices=list(range(len(feats))),
            model_name="TEST", iteration=0,
        )

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        reloaded = GPRExpert.from_checkpoint(ckpt, str(feats_path), str(labels_path))

    after = reloaded.predict(test_feats)
    for k in before:
        np.testing.assert_allclose(after[k], before[k], rtol=1e-5, atol=1e-6)


# ---------- MoEBundle validation ----------

def test_bundle_from_components_accepts_consistent_inputs():
    bundle = _build_moe_bundle(seed=0)
    assert bundle.label_scaler_scope == "all"
    assert bundle.transform == "yeoj"


def test_bundle_rejects_transform_mismatch():
    """If one expert was trained on transform='log' but the bundle says yeoj, fail loudly."""
    feats = _make_raw_features_df(20, seed=3)
    labels = _make_synthetic_labels(feats, seed=3)
    is_ps = (labels["exp_density"] > 0).to_numpy().astype(int)
    scaler1, scaler2 = _fit_scalers(labels, "yeoj")

    ps_ex = _train_tiny_expert(feats, labels, scaler1, scaler2, transform="yeoj")
    nps_ex = _train_tiny_expert(feats, labels, scaler1, scaler2, transform="log")   # mismatch
    _stamp_provenance(ps_ex, "ps")
    _stamp_provenance(nps_ex, "nonps")
    rf_bundle = _make_rf_bundle(feats, is_ps)

    with pytest.raises(ValueError, match="transform"):
        MoEBundle.from_components(rf_bundle, ps_ex, nps_ex)


def test_bundle_rejects_regime_label_mismatch():
    """An expert stamped 'ps' but loaded into the 'nonps' slot must be flagged."""
    feats = _make_raw_features_df(20, seed=4)
    labels = _make_synthetic_labels(feats, seed=4)
    is_ps = (labels["exp_density"] > 0).to_numpy().astype(int)
    scaler1, scaler2 = _fit_scalers(labels, "yeoj")
    ps_ex = _train_tiny_expert(feats, labels, scaler1, scaler2)
    nps_ex = _train_tiny_expert(feats, labels, scaler1, scaler2)
    _stamp_provenance(ps_ex, "ps")
    _stamp_provenance(nps_ex, "ps")   # wrong slot — stamped 'ps' but loaded as nonps
    rf_bundle = _make_rf_bundle(feats, is_ps)
    with pytest.raises(ValueError, match="regime"):
        MoEBundle.from_components(rf_bundle, ps_ex, nps_ex)


# ---------- MoESurrogate ----------

def test_moe_surrogate_rejects_regime_scope():
    """Z-space mixing is only valid under scope='all'. Construction must reject 'regime'."""
    bundle = _build_moe_bundle(seed=5, scope="regime")
    with pytest.raises(ValueError, match="scope='all'"):
        MoESurrogate(bundle)


def test_moe_surrogate_predict_pool_shape_and_supports_joint_sampling():
    bundle = _build_moe_bundle(seed=6)
    sur = MoESurrogate(bundle)
    assert sur.supports_joint_sampling is True

    test_feats = _make_raw_features_df(11, seed=66)
    pool = sur.predict_pool(test_feats)
    assert pool.means.shape == (11, 2)
    assert pool.stds.shape == (11, 2)
    assert np.isfinite(pool.means).all()
    assert (pool.stds >= 0).all()


def test_moe_pool_posterior_soft_means_match_combine_soft():
    """
    Soft policy: pool.means[i] == combine_soft(p_ps[i], mu_ps[i], mu_nps[i]).
    Pins the combine path against the analytic identity, so future refactors of
    `MoEPoolPosterior` cannot silently swap policies.
    """
    bundle = _build_moe_bundle(seed=7)
    sur = MoESurrogate(bundle, policy="soft")
    test_feats = _make_raw_features_df(8, seed=77)
    pool = sur.predict_pool(test_feats)
    assert isinstance(pool, MoEPoolPosterior)

    # Recompute the blend from the underlying expert posteriors + gate.
    ps_pred = bundle.ps_expert.predict(test_feats)
    nps_pred = bundle.nonps_expert.predict(test_feats)
    X_rf, _ = build_rf_features(test_feats, bundle.rf_raw_feature_columns,
                                bundle.rf_converted_feature_columns)
    p_ps = classifier_p_ps(bundle.rf, X_rf)

    exp_means_obj0 = combine_soft(p_ps, ps_pred["exp_density_z_mean"], nps_pred["exp_density_z_mean"])
    exp_means_obj1 = combine_soft(p_ps, ps_pred["diff_z_mean"], nps_pred["diff_z_mean"])
    np.testing.assert_allclose(pool.means[:, 0], exp_means_obj0, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(pool.means[:, 1], exp_means_obj1, rtol=1e-5, atol=1e-6)


def test_moe_pool_posterior_hard_means_match_combine_hard():
    """Hard policy: pool.means[i] == ps_means if p_ps[i] >= threshold else nps_means."""
    bundle = _build_moe_bundle(seed=8)
    threshold = 0.5
    sur = MoESurrogate(bundle, policy="hard", threshold=threshold)
    test_feats = _make_raw_features_df(10, seed=88)
    pool = sur.predict_pool(test_feats)

    ps_pred = bundle.ps_expert.predict(test_feats)
    nps_pred = bundle.nonps_expert.predict(test_feats)
    X_rf, _ = build_rf_features(test_feats, bundle.rf_raw_feature_columns,
                                bundle.rf_converted_feature_columns)
    p_ps = classifier_p_ps(bundle.rf, X_rf)
    expected_obj0 = combine_hard(p_ps, ps_pred["exp_density_z_mean"], nps_pred["exp_density_z_mean"], threshold)
    expected_obj1 = combine_hard(p_ps, ps_pred["diff_z_mean"], nps_pred["diff_z_mean"], threshold)
    np.testing.assert_allclose(pool.means[:, 0], expected_obj0, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(pool.means[:, 1], expected_obj1, rtol=1e-5, atol=1e-6)


def test_moe_sample_shape_and_dtype():
    """sample(n) -> (n, B, 2) float32 — the MC-EHVI consumer's contract."""
    bundle = _build_moe_bundle(seed=9)
    sur = MoESurrogate(bundle)
    test_feats = _make_raw_features_df(4, seed=99)
    pool = sur.predict_pool(test_feats)
    samples = pool.sample(6)
    assert samples.shape == (6, 4, 2)
    assert samples.dtype == torch.float32


def test_moe_soft_sample_mean_converges_to_blended_mean():
    """
    Under soft policy, the MC mean over a large sample count must converge to
    the analytic gate-weighted mean. This is the key correctness check: it
    proves the per-draw Bernoulli routing in MoEPoolPosterior.sample matches
    the closed-form mean used by analytic EHVI.
    """
    bundle = _build_moe_bundle(seed=10)
    sur = MoESurrogate(bundle, policy="soft")
    test_feats = _make_raw_features_df(3, seed=110)
    pool = sur.predict_pool(test_feats)

    torch.manual_seed(0)
    n = 8000
    samples = pool.sample(n).cpu().numpy()   # (n, B, 2)
    mc_mean = samples.mean(axis=0)            # (B, 2)
    # Generous tolerance — std/sqrt(n) with stds ~O(1) and n=8000 puts the
    # noise floor around 0.01; we want a robust assertion not a tight one.
    np.testing.assert_allclose(mc_mean, pool.means, rtol=0.0, atol=0.1)


def test_moe_hard_sample_routes_per_candidate():
    """
    Under hard policy with thresholds at 0 and 1, every candidate must come
    from one specific expert across all draws — easy to verify because the
    expert posteriors differ.
    """
    bundle = _build_moe_bundle(seed=11)
    test_feats = _make_raw_features_df(4, seed=111)

    # threshold=0: every p_ps >= 0, so always PS expert.
    sur_ps = MoESurrogate(bundle, policy="hard", threshold=0.0)
    pool_ps = sur_ps.predict_pool(test_feats)
    ps_pred = bundle.ps_expert.predict(test_feats)
    np.testing.assert_allclose(pool_ps.means[:, 0], ps_pred["exp_density_z_mean"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(pool_ps.means[:, 1], ps_pred["diff_z_mean"], rtol=1e-5, atol=1e-6)

    # threshold=1.0+eps: every p_ps < threshold, so always nonPS expert.
    sur_nps = MoESurrogate(bundle, policy="hard", threshold=1.0 + 1e-9)
    pool_nps = sur_nps.predict_pool(test_feats)
    nps_pred = bundle.nonps_expert.predict(test_feats)
    np.testing.assert_allclose(pool_nps.means[:, 0], nps_pred["exp_density_z_mean"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(pool_nps.means[:, 1], nps_pred["diff_z_mean"], rtol=1e-5, atol=1e-6)


def test_moe_surrogate_rejects_unknown_policy():
    bundle = _build_moe_bundle(seed=12)
    with pytest.raises(ValueError, match="policy"):
        MoESurrogate(bundle, policy="medium")  # type: ignore[arg-type]
