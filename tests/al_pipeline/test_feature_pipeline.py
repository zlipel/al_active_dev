"""
Tests for the DataFrame-based feature pipeline used by MoE per-expert training.

The split into convert / fit / apply is behavior-preserving: a round trip
through the new DataFrame functions must produce the same normalized output
as the existing numpy-array `convert_and_normalize_features`. This is the
test that guards that contract.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from al_pipeline.data_prep.data_loading import (
    convert_and_normalize_features,
    convert_features,
    fit_feature_normalizer,
    apply_feature_normalizer,
)


AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
FEATURE_COLUMNS = AMINO_ACIDS + [
    "length", "SCD", "SHD", "|net charge|", "sum lambda",
    "beads(+)", "beads(-)", "shan ent", "mol wt",
]


def _make_synthetic_features(n_seqs: int = 50, seed: int = 0) -> pd.DataFrame:
    """
    Build a realistic-shaped synthetic feature DataFrame.

    Sequence lengths drawn from [20, 160] (the project's actual IDP range), AA
    counts that sum (approximately) to length per row, and physically plausible
    ranges for the engineered features.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_seqs):
        L = int(rng.integers(20, 160 + 1))
        # AA counts: draw from a multinomial so they sum to L
        probs = rng.dirichlet(np.ones(20))
        counts = rng.multinomial(L, probs).astype(float)
        # Sample plausible aggregate features
        scd = float(rng.normal(0.0, 0.5))
        shd = float(rng.uniform(0.0, 2.5))
        net = float(rng.integers(0, 10))
        sum_lambda = float(rng.uniform(0.0, 3.5))
        beads_pos = float(rng.integers(0, max(1, L // 5)))
        beads_neg = float(rng.integers(0, max(1, L // 5)))
        shan_ent = float(rng.uniform(2.0, 4.5))
        mol_wt = float(L * 110.0 + rng.normal(0.0, 50.0))
        rows.append(list(counts) + [
            L, scd, shd, net, sum_lambda, beads_pos, beads_neg, shan_ent, mol_wt,
        ])
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


# ---------- DataFrame path matches numpy path ----------

def test_dataframe_path_matches_numpy_path():
    """
    Round-trip the SAME data through both pipelines and assert the normalized
    outputs match. This is the core behavior-preservation guarantee.
    """
    df = _make_synthetic_features(n_seqs=80)

    # Numpy path (existing)
    X = df.to_numpy(dtype=np.float32, copy=True)
    X_norm_np, _stats_np = convert_and_normalize_features(X, train=True)

    # DataFrame path (new)
    df_conv = convert_features(df)
    stats_df = fit_feature_normalizer(df_conv)
    df_norm = apply_feature_normalizer(df_conv, stats_df)
    X_norm_df = df_norm.to_numpy(dtype=np.float32)

    # Same column order, same values. Tolerance is float32 noise from the
    # numpy path's `.astype(np.float32)` and the explicit dtype here.
    # Both paths are float32 throughout; differences should be at the level
    # of summation-order rounding only.
    np.testing.assert_allclose(X_norm_np, X_norm_df, rtol=1e-5, atol=1e-6)


def test_apply_with_saved_stats_matches_fit_and_apply():
    """
    Fit stats on a train subset; apply those stats to a held-out subset; the
    held-out output must match what `apply_feature_normalizer` produces given
    the same stats. Pins the train/test workflow shape.
    """
    df = _make_synthetic_features(n_seqs=120, seed=1)
    df_train = df.iloc[:80].reset_index(drop=True)
    df_test  = df.iloc[80:].reset_index(drop=True)

    # Fit on train, apply to test
    train_conv = convert_features(df_train)
    stats      = fit_feature_normalizer(train_conv)
    test_conv  = convert_features(df_test)
    test_norm_via_apply = apply_feature_normalizer(test_conv, stats)

    # Numpy path: fit on train, apply stats to test
    X_train = df_train.to_numpy(dtype=np.float32, copy=True)
    X_test  = df_test.to_numpy(dtype=np.float32, copy=True)
    _, stats_np = convert_and_normalize_features(X_train, train=True)
    X_test_norm_np = convert_and_normalize_features(X_test, train=False, stats=stats_np)

    np.testing.assert_allclose(
        X_test_norm_np, test_norm_via_apply.to_numpy(dtype=np.float32),
        rtol=1e-5, atol=1e-6,
    )


# ---------- convert_features ----------

def test_convert_features_does_not_mutate_input():
    df = _make_synthetic_features(n_seqs=10)
    df_original = df.copy()
    _ = convert_features(df)
    pd.testing.assert_frame_equal(df, df_original)


def test_convert_features_length_normalizes_aa_counts():
    """After conversion, the 20 AA columns should sum to 1.0 per row (since
    the raw counts summed to length). float32-aware tolerance."""
    df = _make_synthetic_features(n_seqs=30)
    df_conv = convert_features(df)
    aa_sums = df_conv[AMINO_ACIDS].sum(axis=1)
    np.testing.assert_allclose(aa_sums.to_numpy(), 1.0, rtol=1e-5, atol=1e-6)


def test_convert_features_divides_aggregate_cols_by_length():
    """beads(+), beads(-), |net charge|, sum lambda, mol wt all get / length."""
    df = _make_synthetic_features(n_seqs=10)
    df_conv = convert_features(df)
    for col in ["beads(+)", "beads(-)", "|net charge|", "sum lambda", "mol wt"]:
        # Cast expected to float32 too — convert_features is float32 throughout.
        expected = (df[col] / df["length"]).astype(np.float32)
        np.testing.assert_allclose(df_conv[col].to_numpy(), expected.to_numpy(), rtol=1e-6, atol=1e-7)


# ---------- fit_feature_normalizer ----------

def test_fit_uses_population_std_not_sample_std():
    """ddof=0 — matches convert_and_normalize_features. Generations are the
    population, not a sample of a larger one."""
    df = _make_synthetic_features(n_seqs=50, seed=2)
    df_conv = convert_features(df)
    stats = fit_feature_normalizer(df_conv)

    # Hand-compute population std for one standardized column and compare
    feat = "SCD"
    col = df_conv[feat]
    pop_std = float(np.sqrt(((col - col.mean()) ** 2).mean()))
    assert stats[feat]["std"] == pytest.approx(pop_std, rel=1e-9)


def test_fit_handles_zero_std_column():
    """If a column is constant, std would be 0; fit must guard with std=1.0
    so apply_feature_normalizer doesn't divide by zero."""
    df = _make_synthetic_features(n_seqs=20)
    df["SCD"] = 0.0   # force a constant column
    df_conv = convert_features(df)
    stats = fit_feature_normalizer(df_conv)
    assert stats["SCD"]["std"] == 1.0


def test_fit_handles_zero_range_length():
    """If every sequence has the same length, range would be 0; guard with 1.0."""
    df = _make_synthetic_features(n_seqs=10)
    df["length"] = 50.0   # force same length
    df_conv = convert_features(df)
    stats = fit_feature_normalizer(df_conv)
    assert stats["length"]["range"] == 1.0


# ---------- per-expert workflow (the actual MoE use case) ----------

def test_per_subset_normalizer_can_be_fit_independently():
    """
    Smoke test of the MoE per-expert workflow: fit ONE normalizer for a
    'PS-like' subset and a DIFFERENT one for a 'nonPS-like' subset; both
    work without contaminating the other.
    """
    df = _make_synthetic_features(n_seqs=100, seed=3)
    # Synthetic gate: half the rows are "PS"
    is_ps = np.arange(len(df)) < 50

    df_conv = convert_features(df)

    stats_ps    = fit_feature_normalizer(df_conv.loc[is_ps])
    stats_nonps = fit_feature_normalizer(df_conv.loc[~is_ps])

    # Stats should differ on a feature with cross-subset variance
    assert stats_ps["SCD"]["mean"] != stats_nonps["SCD"]["mean"]

    # Apply each expert's stats to its own subset; check shape + finite output
    norm_ps    = apply_feature_normalizer(df_conv.loc[is_ps], stats_ps)
    norm_nonps = apply_feature_normalizer(df_conv.loc[~is_ps], stats_nonps)
    assert norm_ps.shape == (50, 29)
    assert norm_nonps.shape == (50, 29)
    assert np.isfinite(norm_ps.to_numpy()).all()
    assert np.isfinite(norm_nonps.to_numpy()).all()
