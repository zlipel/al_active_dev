"""
AL forward diagnostic — predictive-accuracy validation for MoE vs. global GPR.

Sister to `al_retrospective.py`. Same iter-walk pattern (train on gens 0..N-1,
evaluate on gen N), but instead of HV under counterfactual re-selection this
module reports per-predictor prediction quality: RMSE, MAE, R², bias, Spearman,
and z-space NLL for six predictors × two objectives × two spaces (z, phys) ×
three splits (all, ps, nonps) × three front_types (all, upper, lower), plus
the RF gate's quality per iter × per front. This is the port of MCSC's
`run_generation_forward_moe_validation.py` into al_active_dev.

Front handling: the trained model is front-agnostic (it sees every training
row regardless of front target). Non-seed generations contribute `cfg.ngen`
upper-front candidates followed by `cfg.ngen` lower-front candidates in
labels_gen{N}.csv row order — so the first half of each gen-M pool is
"upper" and the second half is "lower". ONE forward run covers both fronts;
`cfg.front` is only used to satisfy ALConfig.from_cli's required arg and is
otherwise ignored (see moe_forward_diagnostic CLI).

Framing:
  - Each iter M is a train/test split defined by the completed run's
    `generation` column: train = gens < M, test = gen == M. No leakage.
  - Six predictors evaluated on the test set:
      * `global`       — GlobalGPRSurrogate.predict_pool
      * `ps_expert`    — bundle.ps_expert.predict (native z-space)
      * `nonps_expert` — bundle.nonps_expert.predict
      * `moe_soft`     — MoESurrogate(policy='soft').predict_pool
      * `moe_hard`     — MoESurrogate(policy='hard', cfg.moe_threshold).predict_pool
      * `ps_guarded`   — PS expert on rows where p_ps ≥ cfg.moe_threshold, NaN elsewhere
  - Label scalers are refit from the training slice via
    `_fit_label_scalers`. Under scope='all' this exactly matches the MoE
    bundle's internal scalers and approximates the global GPR's (which
    aren't persisted). Uniform inversion z → physical space across all
    predictors.

Outputs written to `cfg.paths.diagnostic_dir`:
  - `forward_predictions_start{N}.csv`  — long, per iter × row × predictor.
  - `forward_metrics_start{N}.csv`      — long, per iter × predictor × property × space × split.
  - `forward_classifier_start{N}.csv`   — one row per iter with RF gate metrics.
  - `forward_ranking_start{N}.csv`      — one row per predictor with cross-iter aggregates.

Every downstream analysis reads these CSVs; no data is thrown away in-memory.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)

from al_pipeline.core.config import ALConfig
from al_pipeline.diagnostic._common import (
    IterationData,
    _build_global_surrogate,
    _build_moe_surrogates,
    _make_iter_cfg,
    _write_training_slice,
    load_completed_run,
)
from al_pipeline.ga.ga_utils import load_moe_bundle
from al_pipeline.surrogates import (
    GlobalGPRSurrogate, MoEBundle, MoESurrogate,
    build_rf_features, classifier_p_ps,
)
from al_pipeline.training.moe_training import _fit_label_scalers


PREDICTORS_FULL_COVERAGE = ("global", "ps_expert", "nonps_expert", "moe_soft", "moe_hard")
PREDICTORS_GUARDED = ("ps_guarded",)
ALL_PREDICTORS = PREDICTORS_FULL_COVERAGE + PREDICTORS_GUARDED


# ---------- z ↔ physical inversion ----------

def _invert_z_to_phys(z_mean: np.ndarray, scaler1, scaler2, transform: str) -> np.ndarray:
    """
    Invert a (B, 2) z-space mean back to raw physical space.

    Order matches ALConfig: column 0 is obj1 (typically `exp_density`),
    column 1 is obj2 (typically `diff`). scaler2's inverse is followed by
    exp() when `transform == 'log'` to mirror the yeoj/log semantics used
    at training time (see moe_training._prepare_label_array).

    NaN inputs (e.g. ps_guarded on non-approved rows) propagate as NaN.
    """
    if z_mean.ndim != 2 or z_mean.shape[1] != 2:
        raise ValueError(f"z_mean must be shape (B, 2); got {z_mean.shape}")

    out = np.full_like(z_mean, np.nan, dtype=np.float64)
    finite = ~np.isnan(z_mean).any(axis=1)
    if not finite.any():
        return out

    z_ok = z_mean[finite]
    obj1_phys = scaler1.inverse_transform(z_ok[:, [0]]).ravel()
    obj2_pre = scaler2.inverse_transform(z_ok[:, [1]]).ravel()
    obj2_phys = np.exp(obj2_pre) if transform == "log" else obj2_pre

    out[finite, 0] = obj1_phys
    out[finite, 1] = obj2_phys
    return out


def _transform_true_labels_to_z(labels_df: pd.DataFrame, obj1: str, obj2: str,
                                  transform: str, scaler1, scaler2) -> np.ndarray:
    """Ground-truth labels mapped into the same z-space the surrogates predict in."""
    y = labels_df[[obj1, obj2]].to_numpy(dtype=np.float64).copy()
    if transform == "log":
        y[:, 1] = np.log(y[:, 1] + 1e-8)
    y_z = np.empty_like(y)
    y_z[:, 0] = scaler1.transform(y[:, [0]]).ravel()
    y_z[:, 1] = scaler2.transform(y[:, [1]]).ravel()
    return y_z


# ---------- per-predictor prediction gathering ----------

def _predict_per_expert_z(expert, pool_feats_raw_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (means_z, stds_z) shape (B, 2) for one GPRExpert on the pool.

    GPRExpert.predict returns a dict with per-objective z-mean/z-var; assemble
    into arrays matching PoolPosterior's (B, 2) contract.
    """
    out = expert.predict(pool_feats_raw_df)
    means = np.column_stack([out["exp_density_z_mean"], out["diff_z_mean"]])
    stds  = np.column_stack([out["exp_density_std_norm"], out["diff_std_norm"]])
    return means, stds


def _collect_predictions(
    cfg_base: ALConfig,
    bundle: MoEBundle,
    global_sur: GlobalGPRSurrogate,
    moe_soft: MoESurrogate,
    moe_hard: MoESurrogate,
    pool_feats_raw_df: pd.DataFrame,
    p_ps: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Per-predictor (means_z, stds_z) shape-(B, 2) arrays. `ps_guarded` has NaN
    rows where `p_ps < cfg.moe_threshold`.
    """
    threshold = float(cfg_base.moe_threshold)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # Global
    pool = global_sur.predict_pool(pool_feats_raw_df)
    out["global"] = (pool.means.copy(), pool.stds.copy())

    # Per-expert (native z-space)
    ps_m, ps_s = _predict_per_expert_z(bundle.ps_expert, pool_feats_raw_df)
    out["ps_expert"] = (ps_m, ps_s)

    nps_m, nps_s = _predict_per_expert_z(bundle.nonps_expert, pool_feats_raw_df)
    out["nonps_expert"] = (nps_m, nps_s)

    # MoE blends
    pool_soft = moe_soft.predict_pool(pool_feats_raw_df)
    out["moe_soft"] = (pool_soft.means.copy(), pool_soft.stds.copy())

    pool_hard = moe_hard.predict_pool(pool_feats_raw_df)
    out["moe_hard"] = (pool_hard.means.copy(), pool_hard.stds.copy())

    # PS-guarded: PS-expert predictions where p_ps ≥ threshold, NaN otherwise.
    ps_guard_m = ps_m.copy()
    ps_guard_s = ps_s.copy()
    mask = p_ps < threshold
    ps_guard_m[mask] = np.nan
    ps_guard_s[mask] = np.nan
    out["ps_guarded"] = (ps_guard_m, ps_guard_s)

    return out


# ---------- metrics ----------

def _split_indices(labels_raw_df: pd.DataFrame, aux_col: str) -> dict[str, np.ndarray]:
    """Boolean masks: all / ps (aux > 0) / nonps (aux <= 0)."""
    aux = labels_raw_df[aux_col].to_numpy()
    return {
        "all":   np.ones(len(aux), dtype=bool),
        "ps":    aux > 0,
        "nonps": aux <= 0,
    }


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2))) if len(a) > 0 else float("nan")


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b))) if len(a) > 0 else float("nan")


def _bias(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(pred - true)) if len(pred) > 0 else float("nan")


def _r2(pred: np.ndarray, true: np.ndarray) -> float:
    if len(pred) < 2:
        return float("nan")
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _spearman(pred: np.ndarray, true: np.ndarray) -> float:
    if len(pred) < 3:
        return float("nan")
    rho = scipy.stats.spearmanr(pred, true).correlation
    return float(rho) if np.isfinite(rho) else float("nan")


def _nll_z_gaussian(pred_mean: np.ndarray, pred_std: np.ndarray, true: np.ndarray,
                     min_std: float = 1e-6) -> float:
    """Mean per-point Gaussian NLL in z-space. Std floored to avoid inf on collapsed variance."""
    if len(pred_mean) == 0:
        return float("nan")
    sig2 = np.maximum(pred_std, min_std) ** 2
    return float(0.5 * np.mean(np.log(2.0 * np.pi * sig2) + (true - pred_mean) ** 2 / sig2))


def _metrics_row(
    predictor: str, heldout_iter: int, property_name: str, space: str, split: str,
    front_type: str, threshold: float, coverage: float,
    pred_mean: np.ndarray, pred_std: np.ndarray | None, true: np.ndarray,
) -> dict[str, Any]:
    """Build one metrics-CSV row for a (predictor × property × space × split × front_type) cell."""
    return {
        "heldout_iter": heldout_iter,
        "predictor":    predictor,
        "property":     property_name,
        "space":        space,
        "split":        split,
        "front_type":   front_type,
        "threshold":    threshold,
        "coverage":     coverage,
        "n":            int(len(pred_mean)),
        "rmse":         _rmse(pred_mean, true),
        "mae":          _mae(pred_mean, true),
        "bias":         _bias(pred_mean, true),
        "r2":           _r2(pred_mean, true),
        "spearman":     _spearman(pred_mean, true),
        "nll_z":        _nll_z_gaussian(pred_mean, pred_std, true) if (space == "z" and pred_std is not None) else float("nan"),
    }


# ---------- front inference (upper/lower per pool row) ----------

def _infer_front_types(pool_labels_raw: pd.DataFrame) -> np.ndarray:
    """
    Split the iter-M pool into upper / lower halves by row position.

    The AL loop convention (per the user): each non-seed generation contributes
    ngen candidates targeting the upper front, then ngen targeting the lower
    front, in that order. features_gen{N}.csv and labels_gen{N}.csv preserve
    that order. So for a gen-M pool of length L:
        rows [0 : L//2]         → 'upper'
        rows [L//2 : L]         → 'lower'
    Handles odd L gracefully (extra row goes to 'lower'), and single-row pools
    default to 'upper'.
    """
    n = len(pool_labels_raw)
    half = n // 2
    return np.array(["upper"] * half + ["lower"] * (n - half), dtype=object)


# ---------- RF gate metrics ----------

def _classifier_metrics_row(
    heldout_iter: int, front_type: str,
    y_true_ps: np.ndarray, p_ps: np.ndarray, threshold: float,
) -> dict[str, Any]:
    """One-row RF-gate summary for (iter × front_type)."""
    n = len(y_true_ps)
    if n == 0:
        return {"heldout_iter": heldout_iter, "front_type": front_type, "n_candidates": 0}

    y_pred = (p_ps >= threshold).astype(int)

    # ROC-AUC needs both classes present; skip gracefully otherwise.
    unique_classes = set(map(int, np.unique(y_true_ps)))
    roc_auc = (float(roc_auc_score(y_true_ps, p_ps))
                if unique_classes == {0, 1} else float("nan"))

    tn, fp, fn, tp = confusion_matrix(y_true_ps, y_pred, labels=[0, 1]).ravel()
    n_ps = int(y_true_ps.sum())
    n_nonps = int(n - n_ps)
    return {
        "heldout_iter":  heldout_iter,
        "front_type":    front_type,
        "n_candidates":  n,
        "n_ps_true":     n_ps,
        "n_nonps_true":  n_nonps,
        "roc_auc":       roc_auc,
        "ps_recall":     float(recall_score(y_true_ps, y_pred, zero_division=0)),
        "ps_precision":  float(precision_score(y_true_ps, y_pred, zero_division=0)),
        "f1":            float(f1_score(y_true_ps, y_pred, zero_division=0)),
        "nonps_fpr":     float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan"),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


# ---------- ranking summary ----------

def _ranking_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-iter aggregate per predictor in z-space.

    Returns one row per predictor with:
      - macro_mean_RMSE_z: per-objective mean, then averaged across objectives,
        for (split='all', front_type='all')
      - mean_RMSE_z_{ps,nonps}: split-conditional averages across both fronts
      - mean_RMSE_z_{upper,lower}: front-conditional averages across both splits
      - mean_NLL_z_{ps,nonps}
      - coverage_all_z_mean: fraction of test rows the predictor is defined on
    Sorted by macro_mean_RMSE_z ascending (best predictor first).
    """
    z_only = metrics_df[metrics_df["space"] == "z"]
    rows: list[dict[str, Any]] = []
    for predictor, g in z_only.groupby("predictor"):
        # Cell selectors used repeatedly below.
        def cell(split: str, front_type: str) -> pd.DataFrame:
            return g[(g["split"] == split) & (g["front_type"] == front_type)]

        # macro mean = per objective mean, then averaged across objectives.
        macro_rmse = (cell("all", "all")
                       .groupby("property")["rmse"].mean().mean())
        row = {
            "predictor":            predictor,
            "threshold":            float(g["threshold"].dropna().iloc[0]) if g["threshold"].notna().any() else float("nan"),
            "coverage_all_z_mean":  float(cell("all", "all")["coverage"].mean()),
            "macro_mean_RMSE_z":    float(macro_rmse) if pd.notna(macro_rmse) else float("nan"),
            "mean_RMSE_z_ps":       float(cell("ps", "all")["rmse"].mean()),
            "mean_RMSE_z_nonps":    float(cell("nonps", "all")["rmse"].mean()),
            "mean_RMSE_z_upper":    float(cell("all", "upper")["rmse"].mean()),
            "mean_RMSE_z_lower":    float(cell("all", "lower")["rmse"].mean()),
            "mean_NLL_z_ps":        float(cell("ps", "all")["nll_z"].mean()),
            "mean_NLL_z_nonps":     float(cell("nonps", "all")["nll_z"].mean()),
            "policy_class":         ("guarded" if predictor in PREDICTORS_GUARDED else "full_coverage"),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values("macro_mean_RMSE_z", na_position="last").reset_index(drop=True)


# ---------- orchestrator ----------

def run_forward(
    runs_root: Path,
    model: str,
    cfg_base: ALConfig,
    n_iters: int,
    *,
    start_iter: int = 1,
    log=None,
) -> dict[str, Any]:
    """
    Walk iters `start_iter..n_iters`, refit MoE + global on gens 0..M-1, and
    evaluate all six predictors on gen M in both z-space and physical space
    per PS/nonPS split. Also record RF-gate quality per iter and produce a
    cross-iter ranking summary.

    Writes four CSVs to `cfg_base.paths.diagnostic_dir` with a `_start{N}`
    suffix. Returns a summary dict for programmatic consumers (tests, CLI
    plotting).
    """
    log_fn = log.info if log is not None else (lambda msg: None)
    runs_root = Path(runs_root)
    diagnostic_dir = cfg_base.paths.diagnostic_dir
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    obj1, obj2 = cfg_base.obj1, cfg_base.obj2
    aux_col = cfg_base.aux1_obj1  # 'density' by default → the PS-split axis
    threshold = float(cfg_base.moe_threshold)

    log_fn(f"[forward] loading completed run from {runs_root / model}")
    all_data: IterationData = load_completed_run(runs_root, model, n_iters)

    predictions_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    classifier_rows: list[dict[str, Any]] = []

    for M in range(start_iter, n_iters + 1):
        log_fn(f"[forward] iter {M}: training on gens < {M}")
        train_feats, train_labels_raw, _ = all_data.training_slice_before(M)
        pool_feats, pool_labels_raw, _ = all_data.proposal_pool_at(M)

        if len(pool_labels_raw) == 0:
            log_fn(f"[forward] iter {M}: no held-out children, skipping.")
            continue

        # Refit label scalers on the training slice; used for z ↔ phys inversion
        # of ALL predictors (including global, whose fitted scalers aren't
        # persisted at training time).
        scaler1, scaler2 = _fit_label_scalers(train_labels_raw, [obj1, obj2], cfg_base.transform)

        # Train MoE + global inside a per-iter tempdir (matches the retrospective).
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            cfg_moe = _make_iter_cfg(tempdir / "moe", cfg_base, iteration=M - 1, train_model_type="moe")
            _write_training_slice(cfg_moe, train_feats, train_labels_raw)
            moe_surs = _build_moe_surrogates(cfg_moe)
            bundle = load_moe_bundle(cfg_moe, temp=False)

            cfg_global = _make_iter_cfg(tempdir / "global", cfg_base, iteration=M - 1, train_model_type="gpr_multitask")
            _write_training_slice(cfg_global, train_feats, train_labels_raw)
            global_sur = _build_global_surrogate(cfg_global)

        moe_soft = moe_surs["moe_soft"]
        moe_hard = moe_surs["moe_hard"]

        # RF gate: p_ps + ground truth for classifier metrics
        X_rf, _ = build_rf_features(pool_feats, bundle.rf_raw_feature_columns,
                                      bundle.rf_converted_feature_columns)
        p_ps = classifier_p_ps(bundle.rf, X_rf)
        y_true_ps = (pool_labels_raw[aux_col].to_numpy() > 0).astype(int)

        # Per-row front inference. The AL loop's convention: first half of a
        # non-seed generation is upper-front, second half is lower-front. See
        # _infer_front_types. Under this design, ONE forward run covers both
        # fronts — cfg.front is not used for training or evaluation.
        front_types_arr = _infer_front_types(pool_labels_raw)
        front_masks = {
            "all":   np.ones(len(pool_labels_raw), dtype=bool),
            "upper": front_types_arr == "upper",
            "lower": front_types_arr == "lower",
        }
        splits = _split_indices(pool_labels_raw, aux_col)

        # Classifier metrics per front (all + upper + lower). The RF gate
        # predicts the same p_ps for a given row regardless of front — this
        # slices by ground-truth front position so front-shaped gate bias
        # (e.g. "gate does well on upper front but not lower") surfaces.
        for front_name, front_mask in front_masks.items():
            if not front_mask.any():
                continue
            classifier_rows.append(_classifier_metrics_row(
                M, front_name, y_true_ps[front_mask], p_ps[front_mask], threshold,
            ))

        # True labels in both spaces (z uses the refit scalers).
        true_phys = pool_labels_raw[[obj1, obj2]].to_numpy(dtype=np.float64)
        true_z = _transform_true_labels_to_z(pool_labels_raw, obj1, obj2, cfg_base.transform, scaler1, scaler2)

        preds = _collect_predictions(cfg_base, bundle, global_sur, moe_soft, moe_hard, pool_feats, p_ps)

        # Long predictions rows — one per (row × predictor). front_type varies
        # per row (upper for the first half, lower for the second).
        for predictor, (mean_z, std_z) in preds.items():
            mean_phys = _invert_z_to_phys(mean_z, scaler1, scaler2, cfg_base.transform)
            for i in range(len(pool_feats)):
                predictions_rows.append({
                    "heldout_iter":          M,
                    "original_index":        int(pool_labels_raw.index[i]),
                    "front_type":            str(front_types_arr[i]),
                    "predictor":             predictor,
                    "p_ps":                  float(p_ps[i]),
                    "pred_exp_density_z":       float(mean_z[i, 0]),
                    "pred_exp_density_z_std":   float(std_z[i, 0]),
                    "pred_exp_density_phys":    float(mean_phys[i, 0]),
                    "pred_diff_z":              float(mean_z[i, 1]),
                    "pred_diff_z_std":          float(std_z[i, 1]),
                    "pred_diff_phys":           float(mean_phys[i, 1]),
                    "true_exp_density":         float(pool_labels_raw.iloc[i][obj1]),
                    "true_diff":                float(pool_labels_raw.iloc[i][obj2]),
                    "true_density":             float(pool_labels_raw.iloc[i][aux_col]),
                    "true_is_ps":               int(y_true_ps[i]),
                })

        # Long metrics rows — per (predictor × property × space × split × front_type).
        for predictor, (mean_z, std_z) in preds.items():
            mean_phys = _invert_z_to_phys(mean_z, scaler1, scaler2, cfg_base.transform)
            valid_pred = ~np.isnan(mean_z).any(axis=1)
            for prop_idx, prop_name in enumerate((obj1, obj2)):
                for space in ("z", "phys"):
                    pred_arr = (mean_z if space == "z" else mean_phys)[:, prop_idx]
                    std_arr = std_z[:, prop_idx] if space == "z" else None
                    true_arr = (true_z if space == "z" else true_phys)[:, prop_idx]
                    for split_name, split_mask in splits.items():
                        for front_name, front_mask in front_masks.items():
                            cell_mask = split_mask & front_mask
                            n_cell = int(cell_mask.sum())
                            if n_cell == 0:
                                metrics_rows.append(_metrics_row(
                                    predictor, M, prop_name, space, split_name, front_name,
                                    threshold, coverage=0.0,
                                    pred_mean=np.array([]), pred_std=None, true=np.array([]),
                                ))
                                continue
                            eval_mask = cell_mask & valid_pred
                            coverage = float(eval_mask.sum() / n_cell)
                            metrics_rows.append(_metrics_row(
                                predictor, M, prop_name, space, split_name, front_name,
                                threshold, coverage=coverage,
                                pred_mean=pred_arr[eval_mask],
                                pred_std=std_arr[eval_mask] if std_arr is not None else None,
                                true=true_arr[eval_mask],
                            ))

    # DataFrame + write.
    suffix = f"_start{start_iter}"
    predictions_df = pd.DataFrame(predictions_rows)
    metrics_df = pd.DataFrame(metrics_rows)
    classifier_df = pd.DataFrame(classifier_rows)
    ranking_df = _ranking_summary(metrics_df) if len(metrics_df) else pd.DataFrame()

    predictions_path = diagnostic_dir / f"forward_predictions{suffix}.csv"
    metrics_path     = diagnostic_dir / f"forward_metrics{suffix}.csv"
    classifier_path  = diagnostic_dir / f"forward_classifier{suffix}.csv"
    ranking_path     = diagnostic_dir / f"forward_ranking{suffix}.csv"

    predictions_df.to_csv(predictions_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    classifier_df.to_csv(classifier_path, index=False)
    ranking_df.to_csv(ranking_path, index=False)

    log_fn(f"[forward] wrote {predictions_path}")
    log_fn(f"[forward] wrote {metrics_path}")
    log_fn(f"[forward] wrote {classifier_path}")
    log_fn(f"[forward] wrote {ranking_path}")

    return {
        "predictions_df": predictions_df,
        "metrics_df":     metrics_df,
        "classifier_df":  classifier_df,
        "ranking_df":     ranking_df,
        "paths": {
            "predictions": predictions_path,
            "metrics":     metrics_path,
            "classifier":  classifier_path,
            "ranking":     ranking_path,
        },
        "start_iter": start_iter,
        "n_iters":    n_iters,
        "obj1":       obj1,
        "obj2":       obj2,
        # Note: no "front" key. run_forward covers BOTH fronts in a single
        # pass; front is a per-row attribute in predictions_df / metrics_df /
        # classifier_df, not a run-level constant.
    }
