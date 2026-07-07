"""
Tests for the AL MoE regime-OOF diagnostic.

Reuses the synthetic-data helpers from test_moe_training so we don't
duplicate the on-disk features + labels CSV construction. The end-to-end
run trains three OOF folds (n=40 synthetic seqs, k=3) which keeps the
slow test under ~30s.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from al_pipeline.core.config import ALConfig
from al_pipeline.diagnostic.al_regime_oof import (
    HARD_THRESHOLDS, OOF_PREDICTORS, run_regime_oof,
)
from tests.al_pipeline.test_moe_training import (
    FEATURE_COLUMNS, _make_raw_features_df, _make_synthetic_labels_df,
)


def _make_cfg(tmp_path: Path, *, n_seqs: int = 40, iteration: int = 0) -> tuple[ALConfig, int]:
    """ALConfig pointing at tmpdir + write synthetic features/labels CSVs."""
    base = tmp_path / "home"; scratch = tmp_path / "scratch"; db = tmp_path / "db"
    for d in (base, scratch, db):
        d.mkdir(parents=True, exist_ok=True)

    cfg = ALConfig(
        model="TEST_MODEL",
        iteration=iteration,
        front="upper",
        base_path=base,
        scratch_path=scratch,
        db_path=db,
        train_model_type="moe",
        transform="yeoj",
        ehvi_variant="standard",
        exploration_strategy="standard",
        obj1="exp_density",
        obj2="diff",
        epochs=25, patience=3, k_folds=3, learning_rate=0.1,
        ngen=8,
        moe_policy="soft", moe_threshold=0.5,
    )
    p = cfg.paths
    p.iter_scratch_dir.mkdir(parents=True, exist_ok=True)
    p.models_dir.mkdir(parents=True, exist_ok=True)

    feats_df = _make_raw_features_df(n_seqs, seed=0)
    labels_df = _make_synthetic_labels_df(feats_df, seed=0, ps_frac=0.5)
    feats_df.to_csv(p.features_csv, index=False)
    labels_df.to_csv(p.labels_csv, index=False)
    return cfg, n_seqs


@pytest.fixture(scope="module")
def oof_out(tmp_path_factory):
    """Run `run_regime_oof` once, share outputs across tests in the module."""
    tmp_path = tmp_path_factory.mktemp("regime_oof_synthetic")
    cfg, n_seqs = _make_cfg(tmp_path, n_seqs=40)
    torch.manual_seed(0); np.random.seed(0)
    out = run_regime_oof(
        cfg, n_folds=3, ps_threshold=0.5,
        skip_oof=False, skip_final=True,   # skip final training for speed
    )
    return out, cfg, n_seqs


# ---------- oof predictions ----------

def test_oof_predictions_shape(oof_out):
    """One row per MoE-valid sequence in the concatenated OOF frame."""
    out, cfg, _ = oof_out
    df = out["oof_df"]
    # Every fold's held-out rows appear exactly once.
    labels = pd.read_csv(cfg.paths.labels_csv)
    both_labels = labels[[cfg.obj1, cfg.obj2]].notna().all(axis=1)
    density_ok = labels[cfg.aux1_obj1].notna()
    n_moe = int((both_labels & density_ok).sum())
    assert len(df) == n_moe
    assert set(df["fold"].unique()) == {1, 2, 3}


def test_oof_predictions_predictor_columns(oof_out):
    """Every OOF predictor has a physical-space column present."""
    out, cfg, _ = oof_out
    df = out["oof_df"]
    for predictor in OOF_PREDICTORS:
        for prop in (cfg.obj1, cfg.obj2):
            col = f"pred_{prop}_{predictor}"
            assert col in df.columns, f"missing {col}"


def test_oof_predictions_z_columns_present(oof_out):
    """Per-expert + soft mixture z-mean/z-var columns for native-z metrics."""
    out, cfg, _ = oof_out
    df = out["oof_df"]
    for predictor in ("global", "ps_expert", "nonps_expert", "moe_soft"):
        for prop in (cfg.obj1, cfg.obj2):
            for suffix in ("_z_mean", "_z_var"):
                col = f"{predictor}_{prop}{suffix}"
                assert col in df.columns, f"missing {col}"


def test_oof_p_ps_in_unit_interval(oof_out):
    """Calibrated gate returns probabilities in [0, 1]."""
    out, _, _ = oof_out
    p = out["oof_df"]["p_ps"].to_numpy()
    assert ((p >= 0.0) & (p <= 1.0)).all()


def test_oof_no_nan_in_non_guarded_predictions(oof_out):
    """
    Physical predictions for full-coverage predictors (global, experts,
    moe_soft, moe_hard) are always finite. `ps_guarded` is allowed NaN
    at gate<threshold rows.
    """
    out, cfg, _ = oof_out
    df = out["oof_df"]
    for predictor in ("global", "ps_expert", "nonps_expert", "moe_soft", "moe_hard"):
        for prop in (cfg.obj1, cfg.obj2):
            col = f"pred_{prop}_{predictor}"
            assert df[col].notna().all(), f"{col} has NaN"


# ---------- metrics ----------

def test_oof_metrics_columns(oof_out):
    """Metrics table has the expected columns + population."""
    out, _, _ = oof_out
    df = out["metrics_df"]
    for col in ("predictor", "property", "space", "threshold", "split", "coverage",
                 "n", "rmse", "mae", "r2", "bias_pred_minus_true", "spearman"):
        assert col in df.columns, f"missing {col}"
    for col in ("nll_z", "mean_std_z", "resid_std_z",
                 "std_resid_std", "cov1sigma_z", "cov2sigma_z"):
        # These may be NaN in physical rows, but the columns must exist.
        assert col in df.columns, f"missing z-space uncertainty col {col}"


def test_oof_metrics_threshold_sweep_for_gated(oof_out):
    """Gated predictors emit one metrics row per HARD_THRESHOLDS entry."""
    out, cfg, _ = oof_out
    df = out["metrics_df"]
    for predictor in ("moe_hard", "ps_guarded"):
        # count per (property, space, split) — should equal len(HARD_THRESHOLDS)
        rows = df[
            (df["predictor"] == predictor)
            & (df["space"] == "z")
            & (df["split"] == "all")
            & (df["property"] == cfg.obj1)
        ]
        thrs = sorted(rows["threshold"].unique())
        assert thrs == sorted(HARD_THRESHOLDS), f"{predictor}: {thrs}"


def test_oof_metrics_non_gated_have_nan_threshold(oof_out):
    """Non-gated predictors set threshold=NaN in the metrics rows."""
    out, _, _ = oof_out
    df = out["metrics_df"]
    for predictor in ("global", "ps_expert", "nonps_expert", "moe_soft"):
        rows = df[df["predictor"] == predictor]
        assert rows["threshold"].isna().all(), \
            f"{predictor} has non-NaN threshold rows"


def test_oof_metrics_z_space_populates_nll(oof_out):
    """z-space rows populate nll_z; physical rows leave it NaN."""
    out, _, _ = oof_out
    df = out["metrics_df"]
    z_rows = df[df["space"] == "z"]
    # Some z rows may still be NaN if a split had < 2 rows or coverage 0;
    # but the majority should be finite for a healthy synthetic run.
    assert z_rows["nll_z"].notna().any()
    phys_rows = df[df["space"] == "physical"]
    assert phys_rows["nll_z"].isna().all()


def test_oof_summary_columns(oof_out):
    """Summary table sorted by z-space mean RMSE with expected columns."""
    out, cfg, _ = oof_out
    df = out["summary_df"]
    assert "predictor" in df.columns
    assert "threshold" in df.columns
    assert "all_split_mean_RMSE_z" in df.columns
    for prop in (cfg.obj1, cfg.obj2):
        assert f"all_split_{prop}_RMSE_z" in df.columns
        assert f"PS_{prop}_RMSE_z" in df.columns
        assert f"nonPS_{prop}_RMSE_z" in df.columns
    # Sort order: ascending by mean RMSE (NaNs at bottom).
    vals = df["all_split_mean_RMSE_z"].dropna().to_numpy()
    assert (np.diff(vals) >= -1e-12).all()


# ---------- classifier ----------

def test_oof_classifier_row_shape(oof_out):
    """Classifier metrics dict has the expected keys."""
    out, _, _ = oof_out
    clf = out["classifier"]
    for key in ("n", "n_ps", "n_nonps", "threshold", "roc_auc", "pr_auc",
                 "brier", "accuracy", "precision", "recall", "f1",
                 "tp", "fp", "tn", "fn",
                 "mean_p_ps_true_ps", "mean_p_ps_true_nonps"):
        assert key in clf, f"missing classifier key {key}"


# ---------- reproducibility ----------

def test_oof_deterministic_under_fixed_seed(tmp_path):
    """
    Two runs with the same cfg.seed_base + torch seeds produce identical
    OOF predictions.
    """
    cfg1, _ = _make_cfg(tmp_path / "run1", n_seqs=30)
    cfg2, _ = _make_cfg(tmp_path / "run2", n_seqs=30)
    for cfg in (cfg1, cfg2):
        torch.manual_seed(42); np.random.seed(42)
    out1 = run_regime_oof(cfg1, n_folds=3, skip_oof=False, skip_final=True)
    torch.manual_seed(42); np.random.seed(42)
    out2 = run_regime_oof(cfg2, n_folds=3, skip_oof=False, skip_final=True)
    # p_ps is the most sensitive stochastic surface (RF ensemble); check it.
    p1 = out1["oof_df"].sort_values("original_index")["p_ps"].to_numpy()
    p2 = out2["oof_df"].sort_values("original_index")["p_ps"].to_numpy()
    np.testing.assert_allclose(p1, p2, atol=1e-9)


# ---------- CLI plot helpers ----------

def test_regime_oof_plot_helper_writes_pngs(oof_out, tmp_path):
    """`_plot_metric_by_predictor` writes a non-empty PNG for each combo."""
    from al_pipeline.cli.moe_regime_oof import (
        _METRIC_SPECS, _plot_classifier_reliability, _plot_metric_by_predictor,
    )

    out, cfg, _ = oof_out
    for metric_col in _METRIC_SPECS:
        for split in ("all", "PS", "nonPS"):
            path = tmp_path / f"regime_oof_{metric_col}_{split.lower()}.png"
            _plot_metric_by_predictor(
                out["metrics_df"], metric_col, split, cfg.obj1, cfg.obj2, 0.5, path,
            )
            assert path.exists() and path.stat().st_size > 0

    clf_path = tmp_path / "regime_oof_classifier_reliability.png"
    _plot_classifier_reliability(out["classifier"], clf_path)
    assert clf_path.exists() and clf_path.stat().st_size > 0
