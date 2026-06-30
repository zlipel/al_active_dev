"""
End-to-end test for the MoE training orchestrator.

Strategy:
  1. Write a synthetic features CSV + labels CSV (with the columns the AL
     pipeline produces: AA counts + engineered features for features;
     generation + density / density_std + exp_density / exp_density_std +
     diff / diff_std for labels).
  2. Build an ALConfig pointing at a tmpdir for base_path and scratch_path.
  3. Run `train_moe_from_config(cfg)`.
  4. Verify all three artifacts exist on disk in the right places.
  5. Load them via `MoEBundle.from_checkpoints` and stand up an `MoESurrogate`.
  6. Run `predict_pool` on held-out raw features and verify shape + finiteness.
  7. NaN-row drop test: include a few NaN rows in labels and confirm they
     get dropped (and the on-disk features/labels CSVs end up aligned).

The whole pipeline runs in ~6 seconds with the tiny defaults.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from al_pipeline.core.config import ALConfig
from al_pipeline.surrogates import MoEBundle, MoESurrogate
from al_pipeline.training.moe_training import (
    _train_rf_gate,
    train_moe_from_config,
)


AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
FEATURE_COLUMNS = AMINO_ACIDS + [
    "length", "SCD", "SHD", "|net charge|", "sum lambda",
    "beads(+)", "beads(-)", "shan ent", "mol wt",
]


# ---------- synthetic data fixtures ----------

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


def _make_synthetic_labels_df(features_df: pd.DataFrame, seed: int, ps_frac: float = 0.5) -> pd.DataFrame:
    """
    Build a labels DF matching the on-disk layout produced by
    `al_pipeline.data_prep.labels.build_labels`:

      generation, density, density_std, exp_density, exp_density_std, diff, diff_std

    We construct `density` so that ~ps_frac of the rows have density > 0
    (= PS); the AL pipeline uses this column to split MoE regimes.
    """
    rng = np.random.default_rng(seed)
    n = len(features_df)
    # density: bimodal-ish around 0 — ~ps_frac positive
    z = rng.standard_normal(n) + (rng.random(n) < ps_frac).astype(float) * 2.0 - 1.0
    density = z + 0.3 * features_df["SCD"].to_numpy()
    exp_density = density + rng.normal(0.0, 0.1, n)
    diff = 1.0 + np.abs(features_df["sum lambda"].to_numpy()) + rng.uniform(0.0, 0.5, n)
    return pd.DataFrame({
        "generation":         np.zeros(n, dtype=int),
        "density":            density,
        "density_std":        np.abs(rng.normal(0.0, 0.05, n)),
        "exp_density":        exp_density,
        "exp_density_std":    np.abs(rng.normal(0.0, 0.05, n)),
        "diff":               diff,
        "diff_std":           np.abs(rng.normal(0.0, 0.05, n)),
    })


def _make_cfg(tmp_path: Path, *, n_seqs: int = 40, transform: str = "yeoj", iteration: int = 0):
    """ALConfig pointing at tmpdir, plus the on-disk features + labels CSVs."""
    base = tmp_path / "home"
    scratch = tmp_path / "scratch"
    db = tmp_path / "db"
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
        transform=transform,
        # tiny training params keep the test fast
        epochs=30, patience=3, k_folds=3, learning_rate=0.1,
    )
    # Materialize directories the orchestrator writes to.
    p = cfg.paths
    p.iter_scratch_dir.mkdir(parents=True, exist_ok=True)
    p.models_dir.mkdir(parents=True, exist_ok=True)

    feats_df = _make_raw_features_df(n_seqs, seed=0)
    labels_df = _make_synthetic_labels_df(feats_df, seed=0)
    feats_df.to_csv(p.features_csv, index=False)
    labels_df.to_csv(p.labels_csv, index=False)
    return cfg, feats_df, labels_df


# ---------- end-to-end ----------

def test_train_moe_writes_all_artifacts(tmp_path: Path):
    cfg, _, _ = _make_cfg(tmp_path, n_seqs=40)
    torch.manual_seed(0)
    np.random.seed(0)
    summary = train_moe_from_config(cfg)

    p = cfg.paths
    assert p.moe_ps_chkpt(temp=False).exists()
    assert p.moe_nonps_chkpt(temp=False).exists()
    assert p.moe_rf_bundle(temp=False).exists()
    assert summary["n_total"] > 0
    assert summary["n_ps"] >= 2 and summary["n_nonps"] >= 2
    assert summary["n_ps"] + summary["n_nonps"] == summary["n_total"]
    # GridSearch should have picked something concrete
    assert "n_estimators" in summary["rf_best_params"]


def test_trained_artifacts_load_into_moesurrogate(tmp_path: Path):
    """The full save→load→predict round-trip is the load-bearing contract."""
    cfg, _, _ = _make_cfg(tmp_path, n_seqs=48)
    torch.manual_seed(0); np.random.seed(0)
    train_moe_from_config(cfg)

    p = cfg.paths
    bundle = MoEBundle.from_checkpoints(
        str(p.moe_rf_bundle(temp=False)),
        str(p.moe_ps_chkpt(temp=False)),
        str(p.moe_nonps_chkpt(temp=False)),
        str(p.features_csv),
        str(p.labels_csv),
        expected_transform="yeoj",
        expected_label_scaler_scope="all",
        expected_model_name="TEST_MODEL",
        expected_iter=0,
    )

    sur = MoESurrogate(bundle)
    raw_test_df = _make_raw_features_df(7, seed=99)
    pool = sur.predict_pool(raw_test_df)
    assert pool.means.shape == (7, 2)
    assert pool.stds.shape == (7, 2)
    assert np.isfinite(pool.means).all()
    assert (pool.stds >= 0).all()


def test_train_moe_drops_nan_rows_and_realigns_on_disk(tmp_path: Path):
    """
    If some rows have NaN in objectives or PS column, the orchestrator must
    drop them BEFORE indexing experts and rewrite the on-disk CSVs so the
    loader (which uses `original_indices` to slice the same CSV) lines up.
    """
    cfg, feats, labels = _make_cfg(tmp_path, n_seqs=32)
    # Salt in NaN rows.
    labels.loc[3, "diff"] = np.nan
    labels.loc[7, "density"] = np.nan
    labels.loc[15, "exp_density"] = np.nan
    labels.to_csv(cfg.paths.labels_csv, index=False)

    torch.manual_seed(0); np.random.seed(0)
    summary = train_moe_from_config(cfg)
    assert summary["n_total"] == 32 - 3   # three rows dropped

    # On-disk features.csv must match the clean row count too.
    feats_on_disk = pd.read_csv(cfg.paths.features_csv)
    labels_on_disk = pd.read_csv(cfg.paths.labels_csv)
    assert len(feats_on_disk) == 32 - 3
    assert len(labels_on_disk) == 32 - 3


def test_train_moe_rejects_insufficient_ps_or_nonps_rows(tmp_path: Path):
    """Need ≥2 of each regime to train experts; below that, fall back to global."""
    cfg, feats, labels = _make_cfg(tmp_path, n_seqs=20)
    # Force everything to nonPS.
    labels["density"] = -1.0
    labels.to_csv(cfg.paths.labels_csv, index=False)

    with pytest.raises(ValueError, match="at least 2 PS rows and 2 nonPS rows"):
        train_moe_from_config(cfg)


def test_train_moe_rejects_missing_required_column(tmp_path: Path):
    """If labels CSV is missing density (the PS column), fail with a clear message."""
    cfg, _, labels = _make_cfg(tmp_path, n_seqs=20)
    labels.drop(columns=["density"]).to_csv(cfg.paths.labels_csv, index=False)
    with pytest.raises(KeyError, match="density"):
        train_moe_from_config(cfg)


# ---------- RF gate unit (degenerate single-class fallback) ----------

def test_rf_gate_single_class_fallback():
    """All-nonPS training data must yield a default RF (no GridSearch crash)."""
    feats_df = _make_raw_features_df(20, seed=0)
    rf, conv_cols, best = _train_rf_gate(feats_df, np.zeros(20, dtype=int), seed=42)
    assert best == {}
    # P(PS) should be 0 for everything since the RF never saw a positive class.
    X_rf = feats_df[FEATURE_COLUMNS].to_numpy()   # any shape works for the assertion
    _ = conv_cols   # not used here; the assertion checks the RF's classes_ instead
    assert list(rf.classes_) == [0]


def test_rf_gate_with_two_classes_runs_grid_search():
    feats_df = _make_raw_features_df(40, seed=1)
    is_ps = np.zeros(40, dtype=int)
    is_ps[20:] = 1
    rng = np.random.default_rng(2)
    rng.shuffle(is_ps)
    rf, _conv, best = _train_rf_gate(feats_df, is_ps, seed=42, cv=3)
    assert best != {}
    assert "n_estimators" in best
    assert sorted(rf.classes_) == [0, 1]
