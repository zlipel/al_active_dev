"""
Regime-OOF diagnostic + final production-model training.

Sister to `al_forward.py`. The forward diagnostic walks iterations
(train on gens 0..N-1, evaluate on gen N); this module runs stratified
k-fold OOF on a SINGLE iteration's labeled data. Every row in the
MoE-valid slice gets exactly one held-out prediction from a model that
never saw it — a standard cross-validated estimate of generalization on
the full labeled set.

Six predictors per row (matches MCSC's `train_moe_regime_diagnostics.py`):
  - `global`       — GPRExpert trained on all tr_idx rows (both PS + nonPS)
  - `ps_expert`    — GPRExpert trained on tr_idx ∩ PS rows
  - `nonps_expert` — GPRExpert trained on tr_idx ∩ nonPS rows
  - `moe_soft`     — combine_soft(p_ps, ps, nonps) in shared z-space
  - `moe_hard`     — p_ps >= τ ? ps : nonps (threshold sweep at metrics time)
  - `ps_guarded`   — p_ps >= τ ? ps : NaN (threshold sweep at metrics time)

Splits reported per predictor × property × space × threshold:
  - `all`, `PS`, `nonPS`
  - PS density quartiles (low q25 / high q75)
  - PS diff quartiles (low q25 / high q75)
  - p_ps probability bins (7 bins from 0 → 1)

Label-scaler scope: hardcoded to `all` — matches AL production
(`train_moe_from_config` hardcodes shared scalers). This means:
  - All three OOF experts predict into the SAME shared z-space, so
    per-expert z-metrics and MIXED-z metrics (soft/hard) are
    directly comparable.
  - Physical predictions are inverse-scaled from shared z via the shared
    scalers — same as production.

Optional final training (`skip_final=False`) calls existing production
paths: `train_moe_from_config` (writes PS + nonPS + gate to `MOE_*`
artifact paths) and the singletask/multitask GPR path (writes to
`gpr_multitask_chkpt`). Both produced models are what beam search
consumes at deployment; the OOF diagnostic and the final training use
the same iteration's data but different training strategies (single-shot
per-fold vs. k-fold-avg-refit on all data).

Outputs under `cfg.paths.diagnostic_dir`:
  - `regime_oof_predictions_iter{N}.csv`    — one row per (row × 6 predictor cols)
  - `regime_oof_metrics_iter{N}.csv`        — long: predictor × property × space × threshold × split
  - `regime_oof_summary_iter{N}.csv`        — compact AL-ranking wide table
  - `regime_oof_classifier_iter{N}.csv`     — RF gate ROC-AUC / PR-AUC / Brier / confusion
  - `regime_oof_metadata_iter{N}.json`      — provenance
"""
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

from al_pipeline.core.config import ALConfig
from al_pipeline.surrogates import build_rf_features, classifier_p_ps
from al_pipeline.surrogates.gpr_expert import GPRExpert
from al_pipeline.surrogates.moe_combine import (
    combine_soft, soft_mixture_variance,
)
from al_pipeline.training.moe_training import (
    FEATURE_COLUMNS, _fit_label_scalers, _train_rf_gate,
)


# ---------------------------------------------------------------------------
# Constants (mirror MCSC; kept private so we don't leak them into the surrogates surface)
# ---------------------------------------------------------------------------

OOF_PREDICTORS: tuple[str, ...] = (
    "global", "ps_expert", "nonps_expert", "moe_soft", "moe_hard", "ps_guarded",
)
# Gated policies depend on the threshold but not on retraining — sweep them
# at metrics time using cached per-expert predictions + p_ps.
_GATED_PREDICTORS: frozenset[str] = frozenset({"moe_hard", "ps_guarded"})
HARD_THRESHOLDS: tuple[float, ...] = (0.15, 0.30, 0.50, 0.70)

# p_ps probability bins: (label, low inclusive, high exclusive; last bin includes 1.0).
_P_PS_BINS: tuple[tuple[str, float, float], ...] = (
    ("p_ps_0.00_0.05",   0.00, 0.05),
    ("p_ps_0.05_0.15",   0.05, 0.15),
    ("p_ps_0.15_0.30",   0.15, 0.30),
    ("p_ps_0.30_0.50",   0.30, 0.50),
    ("p_ps_0.50_0.70",   0.50, 0.70),
    ("p_ps_0.70_0.90",   0.70, 0.90),
    ("p_ps_0.90_1.00",   0.90, 1.0001),
)


# ---------------------------------------------------------------------------
# Fold: train 3 experts + gate on tr_idx; predict on te_idx
# ---------------------------------------------------------------------------

def _fit_experts_and_gate(
    feats_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    is_ps: np.ndarray,
    tr_idx: np.ndarray,
    label_columns: list[str],
    transform: str,
    scaler1, scaler2,
    seed: int,
    cfg: ALConfig,
    log,
) -> dict[str, Any] | None:
    """
    Fit global + PS + nonPS single-shot experts + calibrated RF gate on
    the rows in `tr_idx`. Returns None if either regime is too small to
    train an expert (regime-OOF requires ≥2 rows per regime; MCSC used 5).

    All three experts share the same label scalers (scope='all'), so
    per-expert z-space predictions are directly comparable.
    """
    idx_ps = tr_idx[is_ps[tr_idx]]
    idx_nonps = tr_idx[~is_ps[tr_idx]]
    min_expert_rows = max(2, cfg.k_folds if cfg.k_folds > 1 else 2)
    if len(idx_ps) < min_expert_rows or len(idx_nonps) < min_expert_rows:
        return None

    tr_feats = feats_df.iloc[tr_idx].reset_index(drop=True)
    tr_labels = labels_df.iloc[tr_idx].reset_index(drop=True)
    ps_feats = feats_df.iloc[idx_ps].reset_index(drop=True)
    ps_labels = labels_df.iloc[idx_ps].reset_index(drop=True)
    nps_feats = feats_df.iloc[idx_nonps].reset_index(drop=True)
    nps_labels = labels_df.iloc[idx_nonps].reset_index(drop=True)

    # Every expert gets its OWN copies of the shared scalers so state doesn't
    # collide across experts if any downstream code mutates them.
    global_expert = GPRExpert.train(
        tr_feats, tr_labels, label_columns, transform,
        copy.deepcopy(scaler1), copy.deepcopy(scaler2),
        FEATURE_COLUMNS, lr=cfg.learning_rate, epochs=cfg.epochs, patience=cfg.patience,
    )
    ps_expert = GPRExpert.train(
        ps_feats, ps_labels, label_columns, transform,
        copy.deepcopy(scaler1), copy.deepcopy(scaler2),
        FEATURE_COLUMNS, lr=cfg.learning_rate, epochs=cfg.epochs, patience=cfg.patience,
    )
    nonps_expert = GPRExpert.train(
        nps_feats, nps_labels, label_columns, transform,
        copy.deepcopy(scaler1), copy.deepcopy(scaler2),
        FEATURE_COLUMNS, lr=cfg.learning_rate, epochs=cfg.epochs, patience=cfg.patience,
    )

    y_rf = is_ps[tr_idx].astype(int)
    rf, conv_cols, _best = _train_rf_gate(
        tr_feats, y_rf, seed=seed,
        calibration_method=cfg.moe_calibration_method, log=log,
    )
    return {
        "global": global_expert,
        "ps":     ps_expert,
        "nonps":  nonps_expert,
        "rf":     rf,
        "rf_converted_feature_columns": conv_cols,
        "idx_ps": idx_ps,
        "idx_nonps": idx_nonps,
    }


def _true_labels_to_z(
    labels_raw_df: pd.DataFrame,
    obj1: str, obj2: str,
    transform: str, scaler1, scaler2,
) -> np.ndarray:
    """Ground-truth labels in shared z-space (all experts see the same z)."""
    y = labels_raw_df[[obj1, obj2]].to_numpy(dtype=np.float64).copy()
    if transform == "log":
        y[:, 1] = np.log(y[:, 1] + 1e-8)
    z = np.empty_like(y)
    z[:, 0] = scaler1.transform(y[:, [0]]).ravel()
    z[:, 1] = scaler2.transform(y[:, [1]]).ravel()
    return z


def _invert_z_to_phys(z_mean: np.ndarray, scaler1, scaler2, transform: str) -> np.ndarray:
    """Inverse-scale a (B, 2) z-mean back to physical space. NaN-safe."""
    out = np.full_like(z_mean, np.nan, dtype=np.float64)
    finite = ~np.isnan(z_mean).any(axis=1)
    if not finite.any():
        return out
    z = z_mean[finite]
    out[finite, 0] = scaler1.inverse_transform(z[:, [0]]).ravel()
    pre = scaler2.inverse_transform(z[:, [1]]).ravel()
    out[finite, 1] = np.exp(pre) if transform == "log" else pre
    return out


def _build_fold_frame(
    bundle: dict[str, Any],
    feats_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    original_indices: list[int],
    te_idx: np.ndarray,
    is_ps: np.ndarray,
    fold_idx: int,
    label_columns: list[str],
    aux_col: str,
    ps_threshold: float,
    scaler1, scaler2,
    transform: str,
) -> pd.DataFrame:
    """
    Predict one held-out fold and build a tidy per-row frame.

    Emits:
      - Physical predictions for every non-gated predictor.
      - Per-expert z-mean/z-var (needed for native-z metrics and to
        reconstruct gated policies at metrics time).
      - Ground-truth labels in physical + shared-z spaces.
      - Fold-level bookkeeping (fold index, original row index, p_ps).
    """
    obj1, obj2 = label_columns
    te_feats = feats_df.iloc[te_idx].reset_index(drop=True)
    te_labels = labels_df.iloc[te_idx].reset_index(drop=True)

    # RF gate on held-out rows
    X_rf, _ = build_rf_features(
        te_feats, FEATURE_COLUMNS, bundle["rf_converted_feature_columns"],
    )
    p_ps = classifier_p_ps(bundle["rf"], X_rf)

    # Per-expert z-space predictions
    def _z_from(expert: GPRExpert) -> tuple[np.ndarray, np.ndarray]:
        out = expert.predict(te_feats)
        m = np.column_stack([out[f"{obj1}_z_mean"], out[f"{obj2}_z_mean"]])
        v = np.column_stack([out[f"{obj1}_z_var"], out[f"{obj2}_z_var"]])
        return m, v

    global_zm, global_zv = _z_from(bundle["global"])
    ps_zm,     ps_zv     = _z_from(bundle["ps"])
    nps_zm,    nps_zv    = _z_from(bundle["nonps"])

    # Physical: inverse-scale from shared z. Same scalers everyone shares.
    global_phys = _invert_z_to_phys(global_zm, scaler1, scaler2, transform)
    ps_phys     = _invert_z_to_phys(ps_zm,     scaler1, scaler2, transform)
    nps_phys    = _invert_z_to_phys(nps_zm,    scaler1, scaler2, transform)

    # Ground truth
    true_phys = te_labels[[obj1, obj2]].to_numpy(dtype=np.float64)
    true_z = _true_labels_to_z(te_labels, obj1, obj2, transform, scaler1, scaler2)
    density = te_labels[aux_col].to_numpy(dtype=np.float64)

    # Soft mixture in shared z (valid under scope='all'; matches MoESurrogate).
    p_col = p_ps[:, None]
    soft_zm = combine_soft(p_col, ps_zm, nps_zm)
    soft_zv = np.clip(
        soft_mixture_variance(p_col, ps_zm, ps_zv, nps_zm, nps_zv), 0.0, None,
    )
    soft_phys = _invert_z_to_phys(soft_zm, scaler1, scaler2, transform)

    # Hard mixture at the DEFAULT threshold (metrics-time sweep will
    # reconstruct at other thresholds from the cached per-expert columns).
    use_ps = (p_ps >= ps_threshold)[:, None]
    hard_phys = np.where(use_ps, ps_phys, nps_phys)

    # ps_guarded at default threshold — PS expert where gated, else NaN.
    guarded_phys = np.where(use_ps, ps_phys, np.nan)

    frame = {
        "original_index":            [original_indices[i] for i in te_idx],
        "fold":                      fold_idx,
        "true_is_ps":                is_ps[te_idx].astype(int),
        "density":                   density,
        "p_ps":                      p_ps,
        f"true_{obj1}":              true_phys[:, 0],
        f"true_{obj2}":              true_phys[:, 1],
        f"true_{obj1}_z":            true_z[:, 0],
        f"true_{obj2}_z":            true_z[:, 1],
        # global predictor: physical + z
        f"pred_{obj1}_global":       global_phys[:, 0],
        f"pred_{obj2}_global":       global_phys[:, 1],
        f"global_{obj1}_z_mean":     global_zm[:, 0],
        f"global_{obj2}_z_mean":     global_zm[:, 1],
        f"global_{obj1}_z_var":      global_zv[:, 0],
        f"global_{obj2}_z_var":      global_zv[:, 1],
        # ps_expert
        f"pred_{obj1}_ps_expert":    ps_phys[:, 0],
        f"pred_{obj2}_ps_expert":    ps_phys[:, 1],
        f"ps_expert_{obj1}_z_mean":  ps_zm[:, 0],
        f"ps_expert_{obj2}_z_mean":  ps_zm[:, 1],
        f"ps_expert_{obj1}_z_var":   ps_zv[:, 0],
        f"ps_expert_{obj2}_z_var":   ps_zv[:, 1],
        # nonps_expert
        f"pred_{obj1}_nonps_expert": nps_phys[:, 0],
        f"pred_{obj2}_nonps_expert": nps_phys[:, 1],
        f"nonps_expert_{obj1}_z_mean": nps_zm[:, 0],
        f"nonps_expert_{obj2}_z_mean": nps_zm[:, 1],
        f"nonps_expert_{obj1}_z_var":  nps_zv[:, 0],
        f"nonps_expert_{obj2}_z_var":  nps_zv[:, 1],
        # moe_soft: mean is a shared-z blend; std is the moment-matched mixture std
        f"pred_{obj1}_moe_soft":     soft_phys[:, 0],
        f"pred_{obj2}_moe_soft":     soft_phys[:, 1],
        f"moe_soft_{obj1}_z_mean":   soft_zm[:, 0],
        f"moe_soft_{obj2}_z_mean":   soft_zm[:, 1],
        f"moe_soft_{obj1}_z_var":    soft_zv[:, 0],
        f"moe_soft_{obj2}_z_var":    soft_zv[:, 1],
        # moe_hard at default threshold (sweep at metrics time)
        f"pred_{obj1}_moe_hard":     hard_phys[:, 0],
        f"pred_{obj2}_moe_hard":     hard_phys[:, 1],
        # ps_guarded at default threshold (sweep at metrics time)
        f"pred_{obj1}_ps_guarded":   guarded_phys[:, 0],
        f"pred_{obj2}_ps_guarded":   guarded_phys[:, 1],
    }
    return pd.DataFrame(frame)


# ---------------------------------------------------------------------------
# Splits + predictor arrays + metrics
# ---------------------------------------------------------------------------

def _split_masks(oof_df: pd.DataFrame, obj1: str, obj2: str) -> dict[str, np.ndarray]:
    """
    all / PS / nonPS, PS density q25/q75 splits, PS diff q25/q75 splits, and
    p_ps probability bins. Quartiles operate on the PS subset only.
    """
    n = len(oof_df)
    is_ps = oof_df["true_is_ps"].astype(bool).to_numpy()
    masks: dict[str, np.ndarray] = {
        "all":   np.ones(n, dtype=bool),
        "PS":    is_ps,
        "nonPS": ~is_ps,
    }

    ps_pos = np.where(is_ps)[0]
    if len(ps_pos) >= 4:
        for prop, lo_name, hi_name in (
            (obj1, "PS_low_q25",  "PS_high_q75"),
            (obj2, "PS_Dlow_q25", "PS_Dhigh_q75"),
        ):
            vals = oof_df.iloc[ps_pos][f"true_{prop}"].to_numpy(dtype=np.float64)
            q25 = float(np.nanquantile(vals, 0.25))
            q75 = float(np.nanquantile(vals, 0.75))
            lo = np.zeros(n, dtype=bool)
            hi = np.zeros(n, dtype=bool)
            lo[ps_pos] = vals <= q25
            hi[ps_pos] = vals >= q75
            masks[lo_name] = lo
            masks[hi_name] = hi

    p = oof_df["p_ps"].to_numpy(dtype=np.float64)
    for name, lo, hi in _P_PS_BINS:
        masks[name] = (p >= lo) & (p < hi)
    return masks


def _get_predictor_arrays(
    oof_df: pd.DataFrame,
    predictor: str,
    prop: str,
    space: str,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
    """
    Return (true, pred, var) full-length arrays for one predictor cell.

    Physical space: var is None (predictor is a point estimate). Native-z
    space: var is the predictive z-variance (only defined for the three
    experts + moe_soft; gated policies get their z reconstructed at the
    supplied threshold from cached per-expert columns).
    """
    def col(name: str) -> np.ndarray | None:
        return oof_df[name].to_numpy(dtype=np.float64) if name in oof_df.columns else None

    gate = oof_df["p_ps"].to_numpy(dtype=np.float64) >= threshold

    if space == "physical":
        true = col(f"true_{prop}")
        if predictor in ("global", "ps_expert", "nonps_expert", "moe_soft"):
            pred = col(f"pred_{prop}_{predictor}")
        elif predictor == "moe_hard":
            ps_c, nps_c = col(f"pred_{prop}_ps_expert"), col(f"pred_{prop}_nonps_expert")
            if ps_c is None or nps_c is None:
                return None
            pred = np.where(gate, ps_c, nps_c)
        elif predictor == "ps_guarded":
            ps_c = col(f"pred_{prop}_ps_expert")
            if ps_c is None:
                return None
            pred = np.where(gate, ps_c, np.nan)
        else:
            return None
        if true is None or pred is None:
            return None
        return true, pred, None

    # space == "z" (shared z under scope='all')
    true = col(f"true_{prop}_z")
    if predictor in ("global", "ps_expert", "nonps_expert", "moe_soft"):
        pred = col(f"{predictor}_{prop}_z_mean")
        var = col(f"{predictor}_{prop}_z_var")
    elif predictor == "moe_hard":
        ps_zm, nps_zm = col(f"ps_expert_{prop}_z_mean"), col(f"nonps_expert_{prop}_z_mean")
        ps_zv, nps_zv = col(f"ps_expert_{prop}_z_var"),  col(f"nonps_expert_{prop}_z_var")
        if any(a is None for a in (ps_zm, nps_zm, ps_zv, nps_zv)):
            return None
        pred = np.where(gate, ps_zm, nps_zm)
        var = np.where(gate, ps_zv, nps_zv)
    elif predictor == "ps_guarded":
        ps_zm = col(f"ps_expert_{prop}_z_mean")
        ps_zv = col(f"ps_expert_{prop}_z_var")
        if ps_zm is None or ps_zv is None:
            return None
        pred = np.where(gate, ps_zm, np.nan)
        var = np.where(gate, ps_zv, np.nan)
    else:
        return None
    if true is None or pred is None:
        return None
    return true, pred, var


def _regression_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Standard bag: RMSE, MAE, R², bias, Spearman ρ."""
    if len(true) < 2:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"),
                "bias_pred_minus_true": float("nan"), "spearman": float("nan")}
    resid = pred - true
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    if len(true) >= 3:
        rho = scipy.stats.spearmanr(pred, true).correlation
        rho = float(rho) if np.isfinite(rho) else float("nan")
    else:
        rho = float("nan")
    return {
        "rmse": float(np.sqrt(ss_res / len(true))),
        "mae":  float(np.mean(np.abs(resid))),
        "r2":   float(r2),
        "bias_pred_minus_true": float(np.mean(resid)),
        "spearman": rho,
    }


def _uncertainty_metrics(
    resid: np.ndarray, var: np.ndarray, min_var: float = 1e-9,
) -> dict[str, float]:
    """z-space NLL + coverage + resid/std calibration diagnostics."""
    v = np.clip(var, min_var, None)
    std = np.sqrt(v)
    return {
        "nll_z":         float(0.5 * np.mean(np.log(2.0 * np.pi * v) + resid ** 2 / v)),
        "mean_std_z":    float(np.mean(std)),
        "resid_std_z":   float(np.std(resid)),
        "std_resid_std": float(np.std(resid / std)),
        "cov1sigma_z":   float(np.mean(np.abs(resid) <= std)),
        "cov2sigma_z":   float(np.mean(np.abs(resid) <= 2.0 * std)),
    }


def _finite_and_metrics(true, pred, var) -> tuple[int, dict[str, float]]:
    """Filter to finite triples and compute both metric families."""
    finite = np.isfinite(true) & np.isfinite(pred)
    if var is not None:
        finite = finite & np.isfinite(var)
    n = int(finite.sum())
    out: dict[str, float] = {"n": n}
    if n < 2:
        out.update(_regression_metrics(np.array([]), np.array([])))
        if var is not None:
            out.update({"nll_z": float("nan"), "mean_std_z": float("nan"),
                         "resid_std_z": float("nan"), "std_resid_std": float("nan"),
                         "cov1sigma_z": float("nan"), "cov2sigma_z": float("nan")})
        return n, out
    t, p = true[finite], pred[finite]
    out.update(_regression_metrics(t, p))
    if var is not None:
        out.update(_uncertainty_metrics(t - p, var[finite]))
    return n, out


def _make_all_metrics(oof_df: pd.DataFrame, obj1: str, obj2: str, default_threshold: float) -> pd.DataFrame:
    """Long-format metrics: predictor × property × space × threshold × split."""
    masks = _split_masks(oof_df, obj1, obj2)
    rows: list[dict[str, Any]] = []
    for space in ("physical", "z"):
        for prop in (obj1, obj2):
            for predictor in OOF_PREDICTORS:
                thresholds: tuple[float | None, ...] = (
                    tuple(HARD_THRESHOLDS) if predictor in _GATED_PREDICTORS else (None,)
                )
                for thr in thresholds:
                    use_thr = default_threshold if thr is None else float(thr)
                    arrays = _get_predictor_arrays(oof_df, predictor, prop, space, use_thr)
                    if arrays is None:
                        continue
                    true, pred, var = arrays
                    for split_name, mask in masks.items():
                        total = int(mask.sum())
                        if total == 0:
                            continue
                        t = true[mask]
                        p = pred[mask]
                        v = var[mask] if var is not None else None
                        fin = np.isfinite(t) & np.isfinite(p)
                        if v is not None:
                            fin = fin & np.isfinite(v)
                        coverage = float(fin.sum()) / total
                        _, met = _finite_and_metrics(t, p, v)
                        rows.append({
                            "predictor": predictor, "property": prop, "space": space,
                            "threshold": float("nan") if thr is None else float(thr),
                            "split": split_name, "coverage": coverage, **met,
                        })
    return pd.DataFrame(rows)


def _build_summary_table(metrics_df: pd.DataFrame, obj1: str, obj2: str) -> pd.DataFrame:
    """
    Compact wide table: one row per (predictor, threshold). Primary sort key
    is all-split mean z-RMSE across both objectives (lower = better).
    """
    def _val(predictor: str, thr: float, space: str, prop: str, split: str, metric: str) -> float:
        q = metrics_df[
            (metrics_df["predictor"] == predictor) & (metrics_df["space"] == space)
            & (metrics_df["property"] == prop) & (metrics_df["split"] == split)
        ]
        if predictor in _GATED_PREDICTORS:
            q = q[q["threshold"] == thr]
        else:
            q = q[q["threshold"].isna()]
        if len(q) == 0 or metric not in q.columns:
            return float("nan")
        return float(q.iloc[0][metric])

    combos: list[tuple[str, float]] = []
    for predictor in OOF_PREDICTORS:
        if predictor in _GATED_PREDICTORS:
            combos.extend((predictor, float(t)) for t in HARD_THRESHOLDS)
        else:
            combos.append((predictor, float("nan")))

    def _mean_skipnan(*vals: float) -> float:
        finite = [v for v in vals if v == v]
        return float(np.mean(finite)) if finite else float("nan")

    rows: list[dict[str, Any]] = []
    for predictor, thr in combos:
        ed_rmse = _val(predictor, thr, "z", obj1, "all", "rmse")
        df_rmse = _val(predictor, thr, "z", obj2, "all", "rmse")
        ed_nll  = _val(predictor, thr, "z", obj1, "all", "nll_z")
        df_nll  = _val(predictor, thr, "z", obj2, "all", "nll_z")
        rows.append({
            "predictor": predictor,
            "threshold": thr,
            f"all_split_{obj1}_RMSE_z": ed_rmse,
            f"all_split_{obj2}_RMSE_z": df_rmse,
            "all_split_mean_RMSE_z":    _mean_skipnan(ed_rmse, df_rmse),
            f"all_split_{obj1}_NLL_z":  ed_nll,
            f"all_split_{obj2}_NLL_z":  df_nll,
            "all_split_mean_NLL_z":     _mean_skipnan(ed_nll, df_nll),
            f"PS_{obj1}_RMSE_z":        _val(predictor, thr, "z", obj1, "PS", "rmse"),
            f"PS_{obj2}_RMSE_z":        _val(predictor, thr, "z", obj2, "PS", "rmse"),
            f"nonPS_{obj1}_RMSE_z":     _val(predictor, thr, "z", obj1, "nonPS", "rmse"),
            f"nonPS_{obj2}_RMSE_z":     _val(predictor, thr, "z", obj2, "nonPS", "rmse"),
            f"physical_all_{obj1}_MAE": _val(predictor, thr, "physical", obj1, "all", "mae"),
            f"physical_all_{obj2}_MAE": _val(predictor, thr, "physical", obj2, "all", "mae"),
        })
    return (
        pd.DataFrame(rows)
        .sort_values("all_split_mean_RMSE_z", na_position="last")
        .reset_index(drop=True)
    )


def _classifier_metrics(oof_df: pd.DataFrame, threshold: float) -> dict[str, Any]:
    """RF gate quality on the OOF-concatenated held-out predictions."""
    y = oof_df["true_is_ps"].astype(int).to_numpy()
    p = oof_df["p_ps"].to_numpy(dtype=np.float64)
    yhat = (p >= threshold).astype(int)
    both = len(np.unique(y)) == 2

    def _safe(fn):
        try:
            return float(fn())
        except Exception:
            return float("nan")

    cm = confusion_matrix(y, yhat, labels=[0, 1])
    tn, fp, fn_c, tp = cm.ravel()
    return {
        "n":         int(len(y)),
        "n_ps":      int((y == 1).sum()),
        "n_nonps":   int((y == 0).sum()),
        "threshold": float(threshold),
        "roc_auc":   _safe(lambda: roc_auc_score(y, p)) if both else float("nan"),
        "pr_auc":    _safe(lambda: average_precision_score(y, p)) if both else float("nan"),
        "brier":     _safe(lambda: brier_score_loss(y, p)),
        "accuracy":  _safe(lambda: accuracy_score(y, yhat)),
        "precision": _safe(lambda: precision_score(y, yhat, zero_division=0)),
        "recall":    _safe(lambda: recall_score(y, yhat, zero_division=0)),
        "f1":        _safe(lambda: f1_score(y, yhat, zero_division=0)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn_c),
        "mean_p_ps_true_ps":    float(p[y == 1].mean()) if (y == 1).any() else float("nan"),
        "mean_p_ps_true_nonps": float(p[y == 0].mean()) if (y == 0).any() else float("nan"),
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_regime_oof(
    cfg: ALConfig,
    *,
    n_folds: int = 5,
    ps_threshold: float = 0.5,
    skip_oof: bool = False,
    skip_final: bool = False,
    log: logging.Logger | None = None,
) -> dict[str, Any]:
    """
    Stratified k-fold OOF diagnostic on cfg.iteration's labeled data +
    optional final-model training.

    OOF fold procedure (per fold):
      1. Stratified k-fold split of MoE-valid rows on `is_ps`.
      2. Fit global + PS + nonPS single-shot GPRExperts + calibrated RF
         gate on tr_idx (shared label scalers).
      3. Predict on te_idx; record physical + z-space per expert +
         mixture columns + p_ps.
    Concatenated OOF frame → metrics_long + summary + classifier metrics.

    Final training (when skip_final=False):
      - PS + nonPS + gate via existing `train_moe_from_config` (uses
        `_train_expert_with_kfold`: k-fold + avg-params + retrain on all).
      - Global GPR via existing multitask training path (`train_from_config`
        with train_model_type='gpr_multitask'). Beam search will consume
        the MoE bundle + this global GPR side-by-side.

    Returns a dict with the four dataframes, metadata, and the output paths.
    """
    log_fn = log.info if log is not None else (lambda *_: None)
    label_columns = [cfg.obj1, cfg.obj2]
    aux_col = cfg.aux1_obj1  # 'density' by default

    p = cfg.paths
    diag_dir = p.diagnostic_dir
    diag_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load current-iter features + labels ---------------------------
    features_df = pd.read_csv(p.features_csv)
    labels_df = pd.read_csv(p.labels_csv)
    needed = label_columns + [aux_col]
    missing = [c for c in needed if c not in labels_df.columns]
    if missing:
        raise KeyError(f"Required columns missing from {p.labels_csv}: {missing}")

    both_labels = labels_df[label_columns].notna().all(axis=1)
    density_ok = labels_df[aux_col].notna()
    moe_mask = both_labels & density_ok

    original_indices = labels_df.index[moe_mask].tolist()
    moe_feats = features_df.loc[moe_mask].reset_index(drop=True)
    moe_labels = labels_df.loc[moe_mask].reset_index(drop=True)

    n_moe = len(moe_feats)
    density_vals = moe_labels[aux_col].to_numpy(dtype=np.float64)
    is_ps = density_vals > 0
    n_ps = int(is_ps.sum())
    n_nonps = int((~is_ps).sum())

    log_fn(f"[regime-oof] model={cfg.model} iter={cfg.iteration} "
           f"n_total={len(labels_df)} n_moe={n_moe} n_ps={n_ps} n_nonps={n_nonps}")

    # Shared label scalers, fit on ALL MoE-valid rows (scope='all', matches
    # production `train_moe_from_config`). Every OOF fold reuses these
    # scalers so the OOF frame's z-space columns are directly comparable.
    scaler1, scaler2 = _fit_label_scalers(moe_labels, label_columns, cfg.transform)

    # -------- OOF diagnostics ------------------------------------------
    predictions_path = diag_dir / f"regime_oof_predictions_iter{cfg.iteration}.csv"
    metrics_path     = diag_dir / f"regime_oof_metrics_iter{cfg.iteration}.csv"
    summary_path     = diag_dir / f"regime_oof_summary_iter{cfg.iteration}.csv"
    classifier_path  = diag_dir / f"regime_oof_classifier_iter{cfg.iteration}.csv"
    metadata_path    = diag_dir / f"regime_oof_metadata_iter{cfg.iteration}.json"

    oof_df = pd.DataFrame()
    metrics_df = pd.DataFrame()
    summary_df = pd.DataFrame()
    classifier_row: dict[str, Any] = {}
    folds_skipped: list[int] = []

    if not skip_oof:
        if n_moe < n_folds:
            raise ValueError(f"n_moe={n_moe} < n_folds={n_folds}; cannot run OOF.")

        # Stratify on is_ps so folds preserve class proportions. Fall back
        # to unstratified if either class is smaller than the fold count.
        if n_ps < n_folds or n_nonps < n_folds:
            log_fn(f"[regime-oof] PS={n_ps} or nonPS={n_nonps} < k={n_folds}; "
                   f"falling back to unstratified KFold.")
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=cfg.seed_base)
            split_iter = kf.split(np.arange(n_moe))
        else:
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cfg.seed_base)
            split_iter = kf.split(np.arange(n_moe), is_ps.astype(int))

        fold_frames: list[pd.DataFrame] = []
        for fold_idx, (tr_idx, te_idx) in enumerate(split_iter, start=1):
            log_fn(f"[regime-oof] fold {fold_idx}/{n_folds}: train={len(tr_idx)} held={len(te_idx)}")
            bundle = _fit_experts_and_gate(
                moe_feats, moe_labels, is_ps, tr_idx, label_columns, cfg.transform,
                scaler1, scaler2, seed=cfg.seed_base + fold_idx, cfg=cfg, log=log,
            )
            if bundle is None:
                log_fn(f"[regime-oof] fold {fold_idx}: too few PS/nonPS train rows; skipping.")
                folds_skipped.append(fold_idx)
                continue
            fold_frames.append(_build_fold_frame(
                bundle, moe_feats, moe_labels, original_indices, te_idx, is_ps,
                fold_idx, label_columns, aux_col, ps_threshold,
                scaler1, scaler2, cfg.transform,
            ))

        if not fold_frames:
            raise RuntimeError("All OOF folds skipped; cannot produce diagnostics.")

        oof_df = pd.concat(fold_frames, ignore_index=True)
        oof_df.to_csv(predictions_path, index=False)
        log_fn(f"[regime-oof] wrote {predictions_path}")

        metrics_df = _make_all_metrics(oof_df, cfg.obj1, cfg.obj2, ps_threshold)
        metrics_df.to_csv(metrics_path, index=False)
        log_fn(f"[regime-oof] wrote {metrics_path}")

        summary_df = _build_summary_table(metrics_df, cfg.obj1, cfg.obj2)
        summary_df.to_csv(summary_path, index=False)
        log_fn(f"[regime-oof] wrote {summary_path}")

        classifier_row = _classifier_metrics(oof_df, ps_threshold)
        classifier_row.update({
            "model_name": cfg.model, "iter": cfg.iteration,
            "transform": cfg.transform,
            "moe_calibration_method": cfg.moe_calibration_method,
        })
        pd.DataFrame([classifier_row]).to_csv(classifier_path, index=False)
        log_fn(f"[regime-oof] wrote {classifier_path}")

        metadata = {
            "model_name":              cfg.model,
            "iteration":               cfg.iteration,
            "transform":               cfg.transform,
            "label_scaler_scope":      "all",
            "ps_definition":           f"is_ps := ({aux_col} > 0); nonPS := ({aux_col} == 0)",
            "ps_threshold":            ps_threshold,
            "n_folds":                 n_folds,
            "cv":                      "stratified_on_is_ps",
            "seed_base":               cfg.seed_base,
            "calibrate_classifier":    cfg.moe_calibration_method,
            "n_total":                 int(len(labels_df)),
            "n_moe":                   n_moe,
            "n_ps":                    n_ps,
            "n_nonps":                 n_nonps,
            "folds_skipped":           folds_skipped,
            "hard_threshold_sweep":    list(HARD_THRESHOLDS),
            "predictors":              list(OOF_PREDICTORS),
            "p_ps_bins":               [{"name": n, "lo": lo, "hi": hi}
                                        for n, lo, hi in _P_PS_BINS],
            "metric_spaces":           ["physical", "z"],
        }
        with metadata_path.open("w") as f:
            json.dump(metadata, f, indent=2)
        log_fn(f"[regime-oof] wrote {metadata_path}")

    # -------- Final production models ----------------------------------
    final_paths: dict[str, str] = {}
    if not skip_final:
        # Deferred import so the diagnostic module doesn't force pulling the
        # training-config graph when a caller only wants the OOF math.
        from al_pipeline.training.kfold_training import train_from_config
        from al_pipeline.training.moe_training import train_moe_from_config

        log_fn("[regime-oof] training final MoE bundle (PS + nonPS + gate) via existing pipeline")
        cfg_moe = _clone_cfg(cfg, train_model_type="moe")
        train_moe_from_config(cfg_moe, log=log)
        final_paths["moe_ps"]    = str(p.moe_ps_chkpt(temp=False))
        final_paths["moe_nonps"] = str(p.moe_nonps_chkpt(temp=False))
        final_paths["moe_rf"]    = str(p.moe_rf_bundle(temp=False))

        log_fn("[regime-oof] training final global multitask GPR via existing pipeline")
        cfg_global = _clone_cfg(cfg, train_model_type="gpr_multitask")
        train_from_config(cfg_global, log=log)
        final_paths["gpr_multitask"] = str(p.gpr_multitask_chkpt(temp=False))

    return {
        "oof_df":         oof_df,
        "metrics_df":     metrics_df,
        "summary_df":     summary_df,
        "classifier":     classifier_row,
        "paths": {
            "predictions":   predictions_path,
            "metrics":       metrics_path,
            "summary":       summary_path,
            "classifier":    classifier_path,
            "metadata":      metadata_path,
            "final":         final_paths,
        },
    }


def _clone_cfg(cfg: ALConfig, *, train_model_type: str) -> ALConfig:
    """Shallow ALConfig clone with an overridden `train_model_type`."""
    from dataclasses import fields, replace
    if "train_model_type" in {f.name for f in fields(ALConfig)}:
        return replace(cfg, train_model_type=train_model_type)
    return cfg  # pragma: no cover — should not happen
