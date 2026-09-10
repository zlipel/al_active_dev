"""
Tests for the MoE training-time fit plots (al_pipeline.diagnostic.moe_fit_plots):

  1. similarity_cluster_holdout — regime-stratified, ~20% dissimilar test,
     disjoint train/test, >=2 train rows/regime, deterministic.
  2. plot_moe_insample_fit — all-data aggregate-policy parity plot writes
     FIT_{obj}.png and returns finite R²/NLPD.
  3. plot_moe_holdout_fit — similarity-holdout parity plot writes HELDOUT_{obj}.png
     and returns finite R²/NLPD.

CSV/DataFrame-only: builds synthetic features/labels and trains tiny GPs; no
SequenceFeaturizer / force-field DB (which don't exist in the test env).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from al_pipeline.core.config import ALConfig
from al_pipeline.diagnostic.moe_fit_plots import (
    plot_global_holdout_fit, plot_moe_holdout_fit, plot_moe_insample_fit,
    similarity_cluster_holdout,
)


AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
FEATURE_COLUMNS = AMINO_ACIDS + [
    "length", "SCD", "SHD", "|net charge|", "sum lambda",
    "beads(+)", "beads(-)", "shan ent", "mol wt",
]


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


def _make_labels_df(features_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(features_df)
    z = rng.standard_normal(n) + (rng.random(n) < 0.5).astype(float) * 2.0 - 1.0
    density = z + 0.3 * features_df["SCD"].to_numpy()
    exp_density = density + rng.normal(0.0, 0.1, n)
    diff = 1.0 + np.abs(features_df["sum lambda"].to_numpy()) + rng.uniform(0.0, 0.5, n)
    return pd.DataFrame({
        "generation":      np.zeros(n, dtype=int),
        "density":         density,
        "density_std":     np.abs(rng.normal(0.0, 0.05, n)),
        "exp_density":     exp_density,
        "exp_density_std": np.abs(rng.normal(0.0, 0.05, n)),
        "diff":            diff,
        "diff_std":        np.abs(rng.normal(0.0, 0.05, n)),
    })


@pytest.fixture(scope="module")
def synth():
    """Synthetic features/labels/is_ps + a cfg with an ensured models_dir."""
    feats = _make_raw_features_df(48, seed=0)
    labels = _make_labels_df(feats, seed=0)
    is_ps = (labels["density"] > 0).to_numpy().astype(int)
    return feats, labels, is_ps


def _make_cfg(tmp_path: Path, policy: str = "soft") -> ALConfig:
    base = tmp_path / "home"; scratch = tmp_path / "scratch"; db = tmp_path / "db"
    for d in (base, scratch, db):
        d.mkdir(parents=True, exist_ok=True)
    cfg = ALConfig(
        model="TEST_MODEL", iteration=0, front="upper",
        base_path=base, scratch_path=scratch, db_path=db,
        train_model_type="moe", transform="yeoj",
        moe_policy=policy, moe_threshold=0.5,
        epochs=20, patience=3, k_folds=2, learning_rate=0.1,
    )
    p = cfg.ensure()
    p.models_dir.mkdir(parents=True, exist_ok=True)
    return cfg


# ---------- (1) similarity split ----------

def test_similarity_cluster_holdout_regime_stratified(synth):
    feats, labels, is_ps = synth
    is_ps_bool = is_ps.astype(bool)

    tr, te = similarity_cluster_holdout(feats, is_ps_bool, test_frac=0.2, seed=0)

    # Disjoint + covering.
    assert set(tr).isdisjoint(set(te))
    assert sorted([*tr, *te]) == list(range(len(feats)))
    # Both regimes present in each side.
    for idx in (tr, te):
        assert is_ps_bool[idx].any() and (~is_ps_bool[idx]).any()
    # >= 2 train rows per regime.
    assert is_ps_bool[tr].sum() >= 2 and (~is_ps_bool[tr]).sum() >= 2
    # Roughly a fifth held out (cluster holdout is approximate).
    assert 0.08 <= len(te) / len(feats) <= 0.40
    # Deterministic under a fixed seed.
    tr2, te2 = similarity_cluster_holdout(feats, is_ps_bool, test_frac=0.2, seed=0)
    np.testing.assert_array_equal(te, te2)
    np.testing.assert_array_equal(tr, tr2)


# ---------- (2) in-sample fit plot ----------

def test_plot_moe_insample_fit_writes_and_returns(tmp_path, synth):
    feats, labels, is_ps = synth
    cfg = _make_cfg(tmp_path, policy="soft")
    obj1, obj2 = cfg.obj1, cfg.obj2

    # Build deployed-style experts + gate via the shared diagnostic helper.
    from al_pipeline.diagnostic.al_regime_oof import _fit_experts_and_gate
    from al_pipeline.training.moe_training import _fit_label_scalers

    torch.manual_seed(0); np.random.seed(0)
    scaler1, scaler2 = _fit_label_scalers(labels, [obj1, obj2], cfg.transform)
    bundle = _fit_experts_and_gate(
        feats, labels, is_ps.astype(bool), np.arange(len(feats)),
        [obj1, obj2], cfg.transform, scaler1, scaler2, cfg.seed_base, cfg, None,
    )
    assert bundle is not None

    out = plot_moe_insample_fit(
        cfg, feats, labels, is_ps,
        bundle["ps"], bundle["nonps"], bundle["rf"],
        bundle["rf_converted_feature_columns"], scaler1, scaler2,
    )

    p = cfg.paths
    for label in (obj1, obj2):
        fig = p.models_dir / f"MoE_soft_iter0_{p.tag}_FIT_{label}.png"
        assert fig.exists()
        assert np.isfinite(out[label]["r2"]) and np.isfinite(out[label]["nll_z"])
        assert out[label]["n"] == len(feats)


# ---------- (3) held-out fit plot ----------

def test_plot_moe_holdout_fit_writes_and_returns(tmp_path, synth):
    feats, labels, is_ps = synth
    cfg = _make_cfg(tmp_path, policy="soft")
    obj1, obj2 = cfg.obj1, cfg.obj2

    torch.manual_seed(0); np.random.seed(0)
    out = plot_moe_holdout_fit(cfg, feats, labels, is_ps)

    assert out, "held-out plot returned empty (split should support both regimes)"
    p = cfg.paths
    for label in (obj1, obj2):
        fig = p.models_dir / f"MoE_soft_iter0_{p.tag}_HELDOUT_{label}.png"
        assert fig.exists()
        assert np.isfinite(out[label]["r2"]) and np.isfinite(out[label]["nll_z"])
        # Held-out test is a strict subset of the data.
        assert 1 <= out[label]["n"] < len(feats)


# ---------- (4) global multitask held-out fit plot ----------

def test_plot_global_holdout_fit_writes_and_returns(tmp_path, synth):
    """Same 80-20 similarity holdout applied to the global multitask GPR: writes
    GPR_multitask_..._HELDOUT_{obj}.png and returns finite R²/NLPD on the test."""
    feats, labels, is_ps = synth
    cfg = _make_cfg(tmp_path, policy="soft")
    obj1, obj2 = cfg.obj1, cfg.obj2

    torch.manual_seed(0); np.random.seed(0)
    out = plot_global_holdout_fit(cfg, feats, labels, is_ps)

    assert out, "global holdout returned empty (split should support both regimes)"
    p = cfg.paths
    for label in (obj1, obj2):
        fig = p.models_dir / f"GPR_multitask_iter0_{p.tag}_HELDOUT_{label}.png"
        assert fig.exists()
        assert np.isfinite(out[label]["r2"]) and np.isfinite(out[label]["nll_z"])
        assert 1 <= out[label]["n"] < len(feats)
