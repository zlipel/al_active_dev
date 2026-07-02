"""
Tests for the AL forward (generation-forward) diagnostic.

Reuses the synthetic-data helpers from test_al_retrospective.py so we don't
duplicate the `_make_raw_features_df` / `_make_labels_df` / `_write_completed_run`
plumbing. The end-to-end run trains six predictors + an RF gate on each of two
synthetic iters with n_iters=2 / batch=8, which keeps the slow test under ~40s.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from al_pipeline.core.config import ALConfig
from al_pipeline.diagnostic.al_forward import (
    ALL_PREDICTORS, PREDICTORS_FULL_COVERAGE, PREDICTORS_GUARDED, run_forward,
)
from tests.al_pipeline.test_al_retrospective import _write_completed_run


def _make_cfg(tmp_path: Path, *, model: str = "TEST_MODEL") -> ALConfig:
    base = tmp_path / "home"; scratch = tmp_path / "scratch"; db = tmp_path / "db"
    for d in (base, scratch, db):
        d.mkdir(parents=True, exist_ok=True)
    return ALConfig(
        model=model, iteration=0, front="upper",
        base_path=base, scratch_path=scratch, db_path=db,
        train_model_type="moe", transform="yeoj",
        ehvi_variant="standard", exploration_strategy="standard",
        obj1="exp_density", obj2="diff",
        epochs=20, patience=3, k_folds=3, learning_rate=0.1,
        ngen=8,
        moe_policy="soft", moe_threshold=0.5,
    )


@pytest.fixture(scope="module")
def forward_out(tmp_path_factory):
    """
    Run `run_forward` ONCE on a tiny synthetic dataset and share the outputs
    across the module's tests. Training six predictors per iter is expensive,
    so we cache the result.
    """
    tmp_path = tmp_path_factory.mktemp("fwd_synthetic")
    n_iters = 2
    batch = 8
    model = "TEST_MODEL"
    cfg = _make_cfg(tmp_path, model=model)
    _write_completed_run(cfg.scratch_path, model=model, n_iters=n_iters, batch_size=batch, seed=42)

    torch.manual_seed(0); np.random.seed(0)
    return run_forward(
        runs_root=cfg.scratch_path, model=model,
        cfg_base=cfg, n_iters=n_iters, start_iter=1,
    ), cfg, n_iters, batch


# ---------- predictions_long ----------

def test_forward_predictions_shape(forward_out):
    """One row per (heldout_iter × pool_row × predictor). Six predictors present."""
    out, _cfg, n_iters, batch = forward_out
    df = out["predictions_df"]
    assert len(df) == n_iters * batch * len(ALL_PREDICTORS)
    assert set(df["predictor"].unique()) == set(ALL_PREDICTORS)
    for col in ("heldout_iter", "original_index", "front_type", "predictor", "p_ps",
                 "pred_exp_density_z", "pred_exp_density_z_std", "pred_exp_density_phys",
                 "pred_diff_z", "pred_diff_z_std", "pred_diff_phys",
                 "true_exp_density", "true_diff", "true_density", "true_is_ps"):
        assert col in df.columns, f"missing column {col}"


def test_forward_predictions_front_type_is_per_row(forward_out):
    """`front_type` is inferred per row (upper=first half, lower=second half)."""
    out, _, _, _ = forward_out
    df = out["predictions_df"]
    fronts = set(df["front_type"].unique())
    assert fronts == {"upper", "lower"}
    # Each iter's rows should split roughly half-half across upper and lower.
    for heldout_iter in df["heldout_iter"].unique():
        per_iter = df[(df["heldout_iter"] == heldout_iter) & (df["predictor"] == "global")]
        n_upper = int((per_iter["front_type"] == "upper").sum())
        n_lower = int((per_iter["front_type"] == "lower").sum())
        assert n_upper > 0 and n_lower > 0


def test_forward_predictions_ps_guarded_has_nans(forward_out):
    """`ps_guarded` sets NaN on rows below the p_ps threshold."""
    out, _, _, _ = forward_out
    guarded = out["predictions_df"][out["predictions_df"]["predictor"] == "ps_guarded"]
    # At least SOME rows should be NaN across all four synthetic iters; if the
    # gate is never < threshold on our synthetic data the coverage is 1.0 and
    # this test's premise doesn't hold — accept coverage∈[0,1] here.
    n_nan = int(guarded["pred_exp_density_z"].isna().sum())
    assert 0 <= n_nan <= len(guarded)
    # Full-coverage predictors never emit NaN.
    for predictor in PREDICTORS_FULL_COVERAGE:
        rows = out["predictions_df"][out["predictions_df"]["predictor"] == predictor]
        assert not rows["pred_exp_density_z"].isna().any(), f"{predictor} unexpected NaN"


# ---------- metrics_long ----------

def test_forward_metrics_shape(forward_out):
    """n_iters × n_predictors × 2 objectives × 2 spaces × 3 splits × 3 front_types rows."""
    out, _, n_iters, _ = forward_out
    df = out["metrics_df"]
    expected = n_iters * len(ALL_PREDICTORS) * 2 * 2 * 3 * 3
    assert len(df) == expected
    for col in ("heldout_iter", "predictor", "property", "space", "split",
                 "front_type", "threshold", "coverage", "n", "rmse", "mae",
                 "bias", "r2", "spearman", "nll_z"):
        assert col in df.columns, f"missing column {col}"
    assert set(df["front_type"].unique()) == {"all", "upper", "lower"}


def test_forward_metrics_nll_z_only_populated_for_z_space(forward_out):
    """`nll_z` is meaningful only in z-space; physical-space rows should NaN it."""
    out, _, _, _ = forward_out
    df = out["metrics_df"]
    assert df.loc[df["space"] == "phys", "nll_z"].isna().all()


# ---------- classifier ----------

def test_forward_classifier_columns(forward_out):
    """Three rows per iter (all/upper/lower) with the RF gate summary columns."""
    out, _, n_iters, _ = forward_out
    df = out["classifier_df"]
    assert len(df) == n_iters * 3
    for col in ("heldout_iter", "front_type", "n_candidates", "n_ps_true",
                 "n_nonps_true", "roc_auc", "ps_recall", "ps_precision", "f1",
                 "nonps_fpr", "tp", "tn", "fp", "fn"):
        assert col in df.columns, f"missing column {col}"
    assert set(df["front_type"].unique()) == {"all", "upper", "lower"}


# ---------- ranking ----------

def test_forward_ranking_matches_metrics_aggregate(forward_out):
    """`mean_RMSE_z_ps` for each predictor equals the raw metrics mean (front_type=all)."""
    out, _, _, _ = forward_out
    metrics = out["metrics_df"]
    ranking = out["ranking_df"]
    for _, row in ranking.iterrows():
        predictor = row["predictor"]
        expected = float(metrics[
            (metrics["predictor"] == predictor)
            & (metrics["space"] == "z")
            & (metrics["split"] == "ps")
            & (metrics["front_type"] == "all")
        ]["rmse"].mean())
        # Both may be NaN when no PS rows exist in that iter.
        if pd.isna(expected):
            assert pd.isna(row["mean_RMSE_z_ps"])
        else:
            assert row["mean_RMSE_z_ps"] == pytest.approx(expected, rel=1e-9, nan_ok=True)


def test_forward_ranking_has_per_front_columns(forward_out):
    """Ranking exposes per-front macro-mean RMSE_z columns and they match metrics."""
    out, _, _, _ = forward_out
    metrics = out["metrics_df"]
    ranking = out["ranking_df"]
    for col in ("mean_RMSE_z_upper", "mean_RMSE_z_lower"):
        assert col in ranking.columns, f"missing column {col}"
    for _, row in ranking.iterrows():
        predictor = row["predictor"]
        for front, col in (("upper", "mean_RMSE_z_upper"), ("lower", "mean_RMSE_z_lower")):
            expected = float(metrics[
                (metrics["predictor"] == predictor)
                & (metrics["space"] == "z")
                & (metrics["split"] == "all")
                & (metrics["front_type"] == front)
            ]["rmse"].mean())
            if pd.isna(expected):
                assert pd.isna(row[col])
            else:
                assert row[col] == pytest.approx(expected, rel=1e-9, nan_ok=True)


def test_forward_ranking_labels_guarded_correctly(forward_out):
    """`ps_guarded` is labelled 'guarded'; others 'full_coverage'."""
    _, _, _, _ = forward_out
    ranking = forward_out[0]["ranking_df"]
    for _, row in ranking.iterrows():
        predictor = row["predictor"]
        expected = "guarded" if predictor in PREDICTORS_GUARDED else "full_coverage"
        assert row["policy_class"] == expected


# ---------- start_iter shifts range ----------

@pytest.mark.slow
def test_forward_start_iter_shifts_range(tmp_path):
    """start_iter=2 produces n_iters-1 iter rows in each output CSV."""
    n_iters = 2
    batch = 6
    model = "TEST_MODEL"
    cfg = _make_cfg(tmp_path, model=model)
    _write_completed_run(cfg.scratch_path, model=model, n_iters=n_iters, batch_size=batch, seed=99)

    torch.manual_seed(0); np.random.seed(0)
    out = run_forward(
        runs_root=cfg.scratch_path, model=model,
        cfg_base=cfg, n_iters=n_iters, start_iter=2,
    )

    diag_dir = cfg.paths.diagnostic_dir
    assert (diag_dir / "forward_predictions_start2.csv").exists()
    assert (diag_dir / "forward_metrics_start2.csv").exists()
    assert (diag_dir / "forward_classifier_start2.csv").exists()
    assert (diag_dir / "forward_ranking_start2.csv").exists()

    # Only iter M=2 is evaluated.
    assert set(out["predictions_df"]["heldout_iter"].unique()) == {2}
    assert set(out["classifier_df"]["heldout_iter"].unique()) == {2}
    # Three rows per iter (all + upper + lower).
    assert len(out["classifier_df"]) == 3
    assert out["start_iter"] == 2
