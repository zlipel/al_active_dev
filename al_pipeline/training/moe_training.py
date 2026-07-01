"""
MoE training orchestrator — per-iter fit of PS + nonPS GPR experts + RF gate.

Called via `train_moe_from_config(cfg)` when `cfg.train_model_type == 'moe'`
(the AL CLI wiring lands in feat/moe-cli-al). Mirrors the per-expert / per-RF
design from [[project-moe-training-plan]]:

  - PS split: row i is PS iff labels_df[cfg.aux1_obj1][i] > 0 (typically
    `density > 0`, i.e. the simulated material phase-separated).
  - Shared label scalers (`scope='all'`): fit ONCE on the full clean training
    set so PS and nonPS experts predict into a common z-space. Mixing in
    z-space is only well-defined under this scope, and `MoESurrogate`
    enforces it at construction time.
  - Per-expert kfold + final fit: each expert mirrors the global GPR's
    "k-fold for warm-start hyperparams → average → final fit on all data"
    pattern from [kfold_training.py]. Same kernel, same trainer, same early
    stopping. Only difference: it operates on the row subset for the
    expert's regime, and uses the SHARED label scalers (not per-fold).
  - RF gate: GridSearchCV over a small grid + refit on all data. Improves
    on the MCSC code (which trained the RF with no hyperparameter search).
    Falls back to a default RF if the training data has only one PS class.

Artifacts written:
  - cfg.paths.moe_ps_chkpt(temp=False)    — PS expert .pt
  - cfg.paths.moe_nonps_chkpt(temp=False) — nonPS expert .pt
  - cfg.paths.moe_rf_bundle(temp=False)   — RF .pkl
  - cfg.paths.features_csv / labels_csv   — UNCHANGED (used as the source of
    truth for ExactGP train tensors when an expert is reloaded)
"""
from __future__ import annotations

import json
from typing import Any

import gpytorch
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import PowerTransformer, StandardScaler

from al_pipeline.core.config import ALConfig
from al_pipeline.data_prep.data_loading import (
    apply_feature_normalizer,
    convert_and_normalize_features,
    convert_features,
    fit_feature_normalizer,
)
from al_pipeline.surrogates import (
    GPRExpert,
    build_rf_features,
    save_rf_bundle,
)
from al_pipeline.surrogates.gpr_expert import _prepare_label_array
from al_pipeline.training.ml_models import MultitaskGPRegressionModel
from al_pipeline.training.trainers import MultitaskGPRTrainer


AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
FEATURE_COLUMNS = AMINO_ACIDS + [
    "length", "SCD", "SHD", "|net charge|", "sum lambda",
    "beads(+)", "beads(-)", "shan ent", "mol wt",
]


# ---------------------------------------------------------------------------
# Label scalers (shared across experts under scope='all')
# ---------------------------------------------------------------------------

def _make_label_scalers(transform: str):
    """Mirror the global pipeline's per-objective scaler choice."""
    if transform == "log":
        # Objective 1 (exp_density) keeps yeoj; objective 2 (diff) is pre-log
        # transformed and then standardized.
        return PowerTransformer(method="yeo-johnson", standardize=True), StandardScaler()
    if transform == "yeoj":
        return (
            PowerTransformer(method="yeo-johnson", standardize=True),
            PowerTransformer(method="yeo-johnson", standardize=True),
        )
    if transform == "none":
        return StandardScaler(), StandardScaler()
    raise ValueError(f"Unknown transform={transform!r}")


def _fit_label_scalers(labels_df: pd.DataFrame, label_columns: list[str], transform: str):
    """Fit shared scalers on the FULL clean training set (scope='all')."""
    y = _prepare_label_array(labels_df, label_columns, transform)
    scaler1, scaler2 = _make_label_scalers(transform)
    scaler1.fit(y[:, [0]])
    scaler2.fit(y[:, [1]])
    return scaler1, scaler2


def _apply_label_scalers(labels_df: pd.DataFrame, label_columns: list[str],
                          transform: str, scaler1, scaler2) -> np.ndarray:
    """Yeoj/log + shared standardization. Returns (N, 2) numpy."""
    y = _prepare_label_array(labels_df, label_columns, transform)
    y_scaled = y.copy()
    y_scaled[:, 0] = scaler1.transform(y[:, [0]]).ravel()
    y_scaled[:, 1] = scaler2.transform(y[:, [1]]).ravel()
    return y_scaled


# ---------------------------------------------------------------------------
# Per-expert training (kfold warm start + final fit)
# ---------------------------------------------------------------------------

def _train_expert_with_kfold(
    features_subset_df: pd.DataFrame,
    labels_subset_df: pd.DataFrame,
    label_columns: list[str],
    transform: str,
    scaler1, scaler2,
    *,
    k_folds: int,
    epochs: int,
    patience: int,
    lr: float,
    log=None,
) -> GPRExpert:
    """
    Train one MoE expert on `features_subset_df` / `labels_subset_df`.

    Pipeline mirrors the global GPR kfold pattern: per-fold feature normalizer,
    per-fold training with early stopping, average state dicts across folds,
    final fit on the full subset with averaged warm start. The ONLY difference
    from the global pipeline is that label scalers are NOT per-fold — they're
    the shared ones passed in (scope='all').
    """
    n = len(features_subset_df)
    # Be safe on tiny subsets — KFold needs at least 2 splits.
    k = max(2, min(k_folds, n // 2))
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    model_dicts: list[dict] = []
    likelihood_dicts: list[dict] = []

    for fold_i, (train_idx, val_idx) in enumerate(kf.split(np.arange(n))):
        train_feats_raw = features_subset_df.iloc[train_idx].reset_index(drop=True)
        val_feats_raw = features_subset_df.iloc[val_idx].reset_index(drop=True)
        train_labels_raw = labels_subset_df.iloc[train_idx].reset_index(drop=True)
        val_labels_raw = labels_subset_df.iloc[val_idx].reset_index(drop=True)

        # Per-fold feature normalizer
        train_conv = convert_features(train_feats_raw[FEATURE_COLUMNS])
        fold_feat_stats = fit_feature_normalizer(train_conv)
        train_norm = apply_feature_normalizer(train_conv, fold_feat_stats)
        val_norm = apply_feature_normalizer(
            convert_features(val_feats_raw[FEATURE_COLUMNS]), fold_feat_stats,
        )

        # SHARED label scalers
        train_y = _apply_label_scalers(train_labels_raw, label_columns, transform, scaler1, scaler2)
        val_y = _apply_label_scalers(val_labels_raw, label_columns, transform, scaler1, scaler2)

        train_x_t = torch.tensor(train_norm.to_numpy(), dtype=torch.float32)
        val_x_t = torch.tensor(val_norm.to_numpy(), dtype=torch.float32)
        train_y_t = torch.tensor(train_y, dtype=torch.float32)
        val_y_t = torch.tensor(val_y, dtype=torch.float32)

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
        model = MultitaskGPRegressionModel(train_x_t, train_y_t, likelihood, num_tasks=2)
        trainer = MultitaskGPRTrainer(
            model, likelihood, learning_rate=lr, epochs=epochs, patience=patience,
        )
        trainer.train((train_x_t, train_y_t), (val_x_t, val_y_t), early_stop=True)
        if log:
            log.info(f"[moe expert kfold {fold_i+1}/{k}] trained")
        model.eval(); likelihood.eval()
        model_dicts.append(model.state_dict())
        likelihood_dicts.append(likelihood.state_dict())

    # Average state dicts (warm start for final fit)
    avg_model_state = {k_: sum(d[k_] for d in model_dicts) / len(model_dicts) for k_ in model_dicts[0]}
    avg_lik_state = {k_: sum(d[k_] for d in likelihood_dicts) / len(likelihood_dicts) for k_ in likelihood_dicts[0]}

    # Final fit on all subset rows
    full_conv = convert_features(features_subset_df[FEATURE_COLUMNS])
    full_feat_stats = fit_feature_normalizer(full_conv)
    full_norm = apply_feature_normalizer(full_conv, full_feat_stats)
    full_y = _apply_label_scalers(labels_subset_df, label_columns, transform, scaler1, scaler2)

    full_x_t = torch.tensor(full_norm.to_numpy(), dtype=torch.float32)
    full_y_t = torch.tensor(full_y, dtype=torch.float32)

    final_lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    final_model = MultitaskGPRegressionModel(full_x_t, full_y_t, final_lik, num_tasks=2)
    final_model.load_state_dict(avg_model_state)
    final_lik.load_state_dict(avg_lik_state)
    final_trainer = MultitaskGPRTrainer(
        final_model, final_lik, learning_rate=lr, epochs=epochs, patience=patience,
    )
    final_trainer.train((full_x_t, full_y_t), None, early_stop=False)
    final_model.eval(); final_lik.eval()
    if log:
        log.info(f"[moe expert final fit] trained on {n} rows")

    return GPRExpert(
        model=final_model,
        likelihood=final_lik,
        feature_normalizer_stats=full_feat_stats,
        label_scaler1=scaler1,
        label_scaler2=scaler2,
        transform=transform,
        label_columns=label_columns,
        feature_columns=FEATURE_COLUMNS,
    )


# ---------------------------------------------------------------------------
# RF gate training
# ---------------------------------------------------------------------------

_RF_PARAM_GRID = {
    "n_estimators":     [50, 100, 200],
    "max_depth":        [None, 8, 16],
    "min_samples_leaf": [1, 3],
}


def _train_rf_gate(
    features_df: pd.DataFrame,
    is_ps: np.ndarray,
    *,
    seed: int = 42,
    cv: int = 5,
    log=None,
) -> tuple[RandomForestClassifier, list[str], dict]:
    """
    CV grid-search the RF, then refit on all data with the best params.

    Single-class degenerate case (only one regime present in training): skip
    CV and fit a default RF. The classifier will return p=0 or p=1 for every
    candidate, which is the correct degenerate behavior — the gate has no
    signal yet.
    """
    X_rf, conv_cols = build_rf_features(features_df, FEATURE_COLUMNS, None)

    if len(np.unique(is_ps)) < 2:
        if log:
            log.warning("Only one PS class in training data; using default RF (no CV).")
        rf = RandomForestClassifier(n_estimators=100, random_state=seed)
        rf.fit(X_rf, is_ps)
        return rf, conv_cols, {}

    # Make sure cv is feasible: each fold needs at least 1 sample of each class.
    min_class = int(min(np.sum(is_ps == 0), np.sum(is_ps == 1)))
    effective_cv = max(2, min(cv, min_class))
    search = GridSearchCV(
        RandomForestClassifier(random_state=seed),
        param_grid=_RF_PARAM_GRID,
        cv=effective_cv,
        scoring="roc_auc",
        n_jobs=1,
        refit=True,
    )
    search.fit(X_rf, is_ps)
    if log:
        log.info(
            f"[moe rf gridsearch] best_params={search.best_params_} "
            f"best_roc_auc={search.best_score_:.4f} (cv={effective_cv})"
        )
    return search.best_estimator_, conv_cols, dict(search.best_params_)


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def train_moe_from_config(cfg: ALConfig, log=None) -> dict[str, Any]:
    """
    Train + persist the MoE artifacts for the current iteration.

    Returns a small dict of training summary stats (counts, RF best params)
    so the caller / CLI can log them; the on-disk artifacts are the source
    of truth for downstream consumers.
    """
    p = cfg.paths
    label_columns = [cfg.obj1, cfg.obj2]
    ps_col = cfg.aux1_obj1

    if log:
        log.info(f"[moe train] iter={cfg.iteration} front={cfg.front} model={cfg.model}")

    features_df = pd.read_csv(p.features_csv)
    labels_df = pd.read_csv(p.labels_csv)

    # Drop rows with NaN in any objective OR the PS column. We do this BEFORE
    # the train/test indexing so original_indices (saved in checkpoints) refer
    # to row positions in the clean frame — which is what the loader will
    # re-derive from the same CSV.
    needed = label_columns + [ps_col]
    missing = [c for c in needed if c not in labels_df.columns]
    if missing:
        raise KeyError(f"Required columns missing from {p.labels_csv}: {missing}")
    clean_idx = labels_df.dropna(subset=needed).index
    features_df = features_df.loc[clean_idx].reset_index(drop=True)
    labels_df = labels_df.loc[clean_idx].reset_index(drop=True)

    # Save the cleaned frames back under the same paths so the loader reads
    # exactly the rows training saw. If the original frame had NaN rows that
    # got dropped here, the on-disk file would have different row offsets than
    # what `original_indices` references. Rewrite the CSVs as the cleaned
    # source of truth.
    features_df.to_csv(p.features_csv, index=False)
    labels_df.to_csv(p.labels_csv, index=False)

    is_ps = (labels_df[ps_col] > 0).to_numpy().astype(int)
    ps_indices = np.flatnonzero(is_ps == 1).tolist()
    nonps_indices = np.flatnonzero(is_ps == 0).tolist()

    if log:
        log.info(f"[moe train] N={len(labels_df)} | PS={len(ps_indices)} | nonPS={len(nonps_indices)}")
    if len(ps_indices) < 2 or len(nonps_indices) < 2:
        raise ValueError(
            f"MoE training needs at least 2 PS rows and 2 nonPS rows; got "
            f"PS={len(ps_indices)} nonPS={len(nonps_indices)}. Fall back to "
            f"the global GPR for this iteration."
        )

    # Shared label scalers (scope='all')
    scaler1, scaler2 = _fit_label_scalers(labels_df, label_columns, cfg.transform)

    # Fit + persist a global feature normalizer too. MoE experts have their
    # own per-regime normalizers (held in each GPRExpert), but the GA's
    # similarity-penalty path and the parents/Pareto-front pipeline both
    # reach into `cfg.paths.norm_stats` and `cfg.paths.features_norm_csv`
    # for globally-normalized data. Writing them here keeps every
    # downstream consumer happy regardless of surrogate type.
    feats_norm_np, global_norm_stats = convert_and_normalize_features(
        features_df[FEATURE_COLUMNS].to_numpy(np.float32), train=True,
    )
    with open(p.norm_stats, "w") as f:
        json.dump(global_norm_stats, f)
    pd.DataFrame(feats_norm_np, columns=FEATURE_COLUMNS).to_csv(p.features_norm_csv, index=False)

    # Labels in shared z-space — same scalers PS / nonPS experts use, so the
    # Pareto-front computation in `get_parents` operates in the same space the
    # MoE surrogate predicts into. scope='all' makes this well-defined.
    labels_scaled = _apply_label_scalers(labels_df, label_columns, cfg.transform, scaler1, scaler2)
    pd.DataFrame(labels_scaled, columns=label_columns).to_csv(p.labels_norm_csv, index=False)

    # PS expert
    if log:
        log.info("[moe train] training PS expert...")
    ps_expert = _train_expert_with_kfold(
        features_df.iloc[ps_indices].reset_index(drop=True),
        labels_df.iloc[ps_indices].reset_index(drop=True),
        label_columns=label_columns,
        transform=cfg.transform,
        scaler1=scaler1, scaler2=scaler2,
        k_folds=cfg.k_folds, epochs=cfg.epochs, patience=cfg.patience, lr=cfg.learning_rate,
        log=log,
    )
    # nonPS expert
    if log:
        log.info("[moe train] training nonPS expert...")
    nonps_expert = _train_expert_with_kfold(
        features_df.iloc[nonps_indices].reset_index(drop=True),
        labels_df.iloc[nonps_indices].reset_index(drop=True),
        label_columns=label_columns,
        transform=cfg.transform,
        scaler1=scaler1, scaler2=scaler2,
        k_folds=cfg.k_folds, epochs=cfg.epochs, patience=cfg.patience, lr=cfg.learning_rate,
        log=log,
    )

    # RF gate
    if log:
        log.info("[moe train] training RF gate...")
    rf, conv_cols, best_params = _train_rf_gate(
        features_df, is_ps, seed=cfg.seed_base, log=log,
    )

    # Save artifacts. Provenance stamps must agree across all three — that's
    # what `MoEBundle._validate_metadata` checks at load time.
    common_provenance = {
        "model_name":         cfg.model,
        "iteration":          cfg.iteration,
        "label_scaler_scope": "all",
    }
    ps_expert.save_checkpoint(
        str(p.moe_ps_chkpt(temp=False)),
        regime="ps",
        original_indices=ps_indices,
        **common_provenance,
    )
    nonps_expert.save_checkpoint(
        str(p.moe_nonps_chkpt(temp=False)),
        regime="nonps",
        original_indices=nonps_indices,
        **common_provenance,
    )
    save_rf_bundle(
        str(p.moe_rf_bundle(temp=False)),
        rf,
        rf_raw_feature_columns=FEATURE_COLUMNS,
        rf_converted_feature_columns=conv_cols,
        ps_definition=f"{ps_col} > 0",
        random_state=cfg.seed_base,
        threshold=0.5,
        model_name=cfg.model,
        iteration=cfg.iteration,
        transform=cfg.transform,
        label_scaler_scope="all",
        best_params=best_params,
    )
    if log:
        log.info(f"[moe train] saved PS={p.moe_ps_chkpt(temp=False)}")
        log.info(f"[moe train] saved nonPS={p.moe_nonps_chkpt(temp=False)}")
        log.info(f"[moe train] saved RF={p.moe_rf_bundle(temp=False)}")

    return {
        "n_total":       len(labels_df),
        "n_ps":          len(ps_indices),
        "n_nonps":       len(nonps_indices),
        "rf_best_params": best_params,
    }
