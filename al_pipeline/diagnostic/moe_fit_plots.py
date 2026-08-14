"""
Training-time surrogate fit plots (MoE + global multitask).

For the MoE, two parity plots per objective, both collapsing the PS / nonPS
experts with the surrogate POLICY (the same combine the AL acquisition uses)
into one aggregate prediction:

  * FIT     — in-sample: the just-trained deployed MoE predicts every training
              row. The MoE analog of the multitask GPR parity plot in
              `kfold_training._kfold_gpr_multitask_from_config`.
  * HELDOUT — an 80-20 split where sequence similarity governs the split
              (regime-stratified cluster holdout). Experts + gate are re-fit on
              the 80% train and the dissimilar 20% test is predicted via the
              policy — a less optimistic "how good are the models given the
              data" estimate.

Everything lives in shared z-space (the space the experts predict into under
``label_scaler_scope='all'`` and the space the multitask plot uses). z-NLPD uses
the mixture variance (law of total variance) so the panels reflect calibration,
not just point accuracy — the same statistic the epsilon shift / pessimism read.

`plot_global_holdout_fit` applies the *same* HELDOUT protocol to the global
multitask GPR (a single GP fit on the 80% train, no gate), so global vs MoE
generalization is directly comparable on an identical split.

Diagnostic only: nothing here mutates the deployed checkpoints. The FIT plot
reuses the just-trained experts; the HELDOUT plots train throwaway models on a
subset. The caller wraps these in ``try/except`` so a plotting failure never
breaks training.
"""
from __future__ import annotations

from typing import Any

import matplotlib
matplotlib.use("Agg")   # headless (cluster)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from al_pipeline.core.config import ALConfig
from al_pipeline.surrogates import (
    build_rf_features, classifier_p_ps, combine_soft, soft_mixture_variance,
)


def _policy_zstats(
    cfg: ALConfig,
    p_ps: np.ndarray,
    ps_zm: np.ndarray, ps_zv: np.ndarray,
    nps_zm: np.ndarray, nps_zv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend per-expert z means/vars into the policy's aggregate (z_mean, z_var).

    Shapes: ``p_ps`` is (B,), the others (B, 2). Soft uses ``combine_soft`` +
    ``soft_mixture_variance`` (law of total variance); hard switches per
    candidate at ``cfg.moe_threshold``. Matches ``MoEPoolPosterior``.
    """
    p_ps = np.asarray(p_ps, dtype=np.float64)
    if cfg.moe_policy == "soft":
        p = p_ps[:, None]
        zm = combine_soft(p, ps_zm, nps_zm)
        zv = np.clip(soft_mixture_variance(p, ps_zm, ps_zv, nps_zm, nps_zv), 0.0, None)
    else:  # hard
        use_ps = (p_ps >= cfg.moe_threshold)[:, None]
        zm = np.where(use_ps, ps_zm, nps_zm)
        zv = np.where(use_ps, ps_zv, nps_zv)
    return zm, zv


def _parity_panels(
    cfg: ALConfig,
    series: list[dict[str, Any]],
    label_columns: list[str],
    kind: str,
    model_label: str,
    log=None,
) -> dict[str, dict[str, float]]:
    """One z-space parity panel per objective, overlaying every series.

    ``model_label`` (e.g. ``"MoE soft"`` / ``"GPR multitask"``) names the plot
    title and the file prefix (spaces -> underscores).

    Each ``series`` entry is a dict::

        {"name": str, "true_z": (N,2), "pred_zm": (N,2), "pred_zv": (N,2),
         "color": str, "alpha": float, "size": float}

    Every series is scattered and annotated (in the legend) with its own R² +
    z-NLPD; the **last** series is the headline (test for HELDOUT, all-data for
    FIT) and its metrics are returned per objective. Styled like the multitask
    parity plot ([kfold_training.py]). Saves
    ``MoE_{policy}_iter{N}_{tag}_{kind}_{label}.png`` to ``models_dir``.
    """
    from al_pipeline.diagnostic.al_regime_oof import _finite_and_metrics  # deferred: import cycle

    p = cfg.paths
    out: dict[str, dict[str, float]] = {}
    for i, label in enumerate(label_columns):
        fig, ax = plt.subplots(figsize=(2.3, 2.3), dpi=300)
        pooled: list[np.ndarray] = []
        headline = {"r2": float("nan"), "nll_z": float("nan"), "n": 0.0}
        for s in series:
            y = np.asarray(s["true_z"][:, i], dtype=np.float64)
            yhat = np.asarray(s["pred_zm"][:, i], dtype=np.float64)
            v = np.asarray(s["pred_zv"][:, i], dtype=np.float64)

            n, met = _finite_and_metrics(y, yhat, v)
            r2 = met.get("r2", float("nan"))
            nll = met.get("nll_z", float("nan"))

            finite = np.isfinite(y) & np.isfinite(yhat)
            ax.scatter(
                y[finite], yhat[finite], color=s["color"], edgecolors="none",
                alpha=s.get("alpha", 0.35), s=s.get("size", 10),
                label=f"{s['name']} (n={n}): $R^2$={r2:.2f}, NLPD={nll:.2f}",
            )
            pooled.append(y[finite]); pooled.append(yhat[finite])
            headline = {"r2": float(r2), "nll_z": float(nll), "n": float(n)}
            if log:
                log.info(f"[{model_label} {kind.lower()}] {label} [{s['name']}]: "
                         f"R2={r2:.4f} NLPD_z={nll:.4f} (n={n})")

        cat = (np.concatenate([a for a in pooled if a.size])
               if any(a.size for a in pooled) else np.array([0.0, 1.0]))
        lo, hi = float(cat.min()), float(cat.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=0.8)
        ax.set_xlabel(f"True {label} (z)", fontsize=6)
        ax.set_ylabel(f"Predicted {label} (z)", fontsize=6)
        ax.set_title(f"{model_label} {kind} — {label}", fontsize=6)
        ax.tick_params(axis="both", which="both", labelsize=4, direction="in")
        ax.legend(fontsize=4.5, loc="best")
        fig.tight_layout()
        prefix = model_label.replace(" ", "_")
        fig_path = p.models_dir / f"{prefix}_iter{cfg.iteration}_{p.tag}_{kind}_{label}.png"
        fig.savefig(str(fig_path), dpi=300, bbox_inches="tight")
        plt.close(fig)

        if log:
            log.info(f"[{model_label} {kind.lower()}] {label}: wrote {fig_path.name}")
        out[label] = headline
    return out


def _policy_from_frame(
    cfg: ALConfig, frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract ``(true_z, pred_zm, pred_zv)`` for the active policy from a fold frame."""
    obj1, obj2 = cfg.obj1, cfg.obj2
    true_z = frame[[f"true_{obj1}_z", f"true_{obj2}_z"]].to_numpy()
    if cfg.moe_policy == "soft":
        zm = frame[[f"moe_soft_{obj1}_z_mean", f"moe_soft_{obj2}_z_mean"]].to_numpy()
        zv = frame[[f"moe_soft_{obj1}_z_var", f"moe_soft_{obj2}_z_var"]].to_numpy()
    else:  # hard: reconstruct z mean/var from per-expert columns at the threshold
        p_ps = frame["p_ps"].to_numpy()
        ps_zm = frame[[f"ps_expert_{obj1}_z_mean", f"ps_expert_{obj2}_z_mean"]].to_numpy()
        ps_zv = frame[[f"ps_expert_{obj1}_z_var", f"ps_expert_{obj2}_z_var"]].to_numpy()
        nps_zm = frame[[f"nonps_expert_{obj1}_z_mean", f"nonps_expert_{obj2}_z_mean"]].to_numpy()
        nps_zv = frame[[f"nonps_expert_{obj1}_z_var", f"nonps_expert_{obj2}_z_var"]].to_numpy()
        zm, zv = _policy_zstats(cfg, p_ps, ps_zm, ps_zv, nps_zm, nps_zv)
    return true_z, zm, zv


def plot_moe_insample_fit(
    cfg: ALConfig,
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    is_ps: np.ndarray,
    ps_expert,
    nonps_expert,
    rf,
    conv_cols: list[str],
    scaler1,
    scaler2,
    *,
    log=None,
) -> dict[str, dict[str, float]]:
    """In-sample aggregate-policy parity plot on all training rows.

    Uses the just-trained deployed experts/gate (passed in — no reload). The
    MoE analog of the multitask GPR in-sample fit plot.
    """
    from al_pipeline.diagnostic.al_regime_oof import _true_labels_to_z  # deferred
    from al_pipeline.training.moe_training import FEATURE_COLUMNS        # deferred

    obj1, obj2 = cfg.obj1, cfg.obj2

    X_rf, _ = build_rf_features(features_df, FEATURE_COLUMNS, conv_cols)
    p_ps = classifier_p_ps(rf, X_rf)

    ps_out = ps_expert.predict(features_df)
    nps_out = nonps_expert.predict(features_df)
    ps_zm = np.column_stack([ps_out[f"{obj1}_z_mean"], ps_out[f"{obj2}_z_mean"]])
    ps_zv = np.column_stack([ps_out[f"{obj1}_z_var"], ps_out[f"{obj2}_z_var"]])
    nps_zm = np.column_stack([nps_out[f"{obj1}_z_mean"], nps_out[f"{obj2}_z_mean"]])
    nps_zv = np.column_stack([nps_out[f"{obj1}_z_var"], nps_out[f"{obj2}_z_var"]])

    zm, zv = _policy_zstats(cfg, np.asarray(p_ps), ps_zm, ps_zv, nps_zm, nps_zv)
    true_z = _true_labels_to_z(labels_df, obj1, obj2, cfg.transform, scaler1, scaler2)
    series = [{
        "name": "all data", "true_z": true_z, "pred_zm": zm, "pred_zv": zv,
        "color": "orange", "alpha": 0.3, "size": 10,
    }]
    return _parity_panels(cfg, series, [obj1, obj2], "FIT", f"MoE {cfg.moe_policy}", log=log)


def _agglomerative_labels(D: np.ndarray, n_clusters: int) -> np.ndarray:
    """Average-linkage agglomerative clustering on a precomputed distance matrix.

    Handles the sklearn ``metric``/``affinity`` rename across versions.
    """
    from sklearn.cluster import AgglomerativeClustering
    try:
        model = AgglomerativeClustering(
            n_clusters=n_clusters, metric="precomputed", linkage="average",
        )
    except TypeError:   # sklearn < 1.2 used `affinity`
        model = AgglomerativeClustering(
            n_clusters=n_clusters, affinity="precomputed", linkage="average",
        )
    return model.fit_predict(D)


def similarity_cluster_holdout(
    features_df: pd.DataFrame,
    is_ps: np.ndarray,
    *,
    test_frac: float = 0.2,
    seed: int = 0,
    min_train_per_regime: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Regime-stratified cluster holdout governed by cosine feature similarity.

    Within each regime separately, cluster rows by cosine distance
    ``D = 0.5 * (1 - cos_sim)`` — the same similarity primitive `ga_utils.alpha`
    uses — then assign whole clusters (smallest first, deterministically) to the
    test set until ~``test_frac`` of that regime is held out, without leaving
    fewer than ``min_train_per_regime`` train rows. Test clusters are absent from
    train, so the holdout is genuinely dissimilar.

    Returns global row indices ``(tr_idx, te_idx)`` into ``features_df``.
    """
    from al_pipeline.training.moe_training import FEATURE_COLUMNS  # deferred

    X = features_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    is_ps = np.asarray(is_ps).astype(bool)
    n_total = len(X)
    test_mask = np.zeros(n_total, dtype=bool)

    for regime_rows in (np.flatnonzero(is_ps), np.flatnonzero(~is_ps)):
        n = len(regime_rows)
        if n < min_train_per_regime + 1:
            continue   # too few rows to spare any for test
        n_test = int(round(test_frac * n))
        n_test = min(n_test, n - min_train_per_regime)
        if n_test < 1:
            continue

        Xr = X[regime_rows]
        norms = np.linalg.norm(Xr, axis=1)
        norms = np.where(norms == 0.0, 1e-12, norms)
        cos = (Xr @ Xr.T) / np.outer(norms, norms)
        D = np.clip(0.5 * (1.0 - cos), 0.0, 1.0)
        np.fill_diagonal(D, 0.0)

        # ~1/test_frac clusters -> one cluster is ~test_frac of the regime, so a
        # few whole clusters land close to the target holdout size.
        n_clusters = int(max(2, min(n - 1, round(1.0 / test_frac))))
        labels = _agglomerative_labels(D, n_clusters)

        cluster_ids, sizes = np.unique(labels, return_counts=True)
        # Smallest clusters first (stable) for a deterministic, near-target holdout.
        order = np.argsort(sizes, kind="stable")
        held = 0
        for ci, csize in zip(cluster_ids[order], sizes[order]):
            if held >= n_test:
                break
            if held + int(csize) > n - min_train_per_regime:
                continue
            test_mask[regime_rows[labels == ci]] = True
            held += int(csize)

    te_idx = np.flatnonzero(test_mask)
    tr_idx = np.flatnonzero(~test_mask)
    return tr_idx, te_idx


def plot_moe_holdout_fit(
    cfg: ALConfig,
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    is_ps: np.ndarray,
    *,
    log=None,
    test_frac: float = 0.2,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Similarity-cluster-holdout aggregate-policy parity plot.

    Re-fits experts + gate on the 80% train (train-only scalers, no leakage),
    predicts the dissimilar 20% test via the policy, and plots the fit. Returns
    ``{}`` (and logs a warning) if the split can't support both regimes.
    """
    from al_pipeline.diagnostic.al_regime_oof import _build_fold_frame, _fit_experts_and_gate
    from al_pipeline.training.moe_training import _fit_label_scalers  # deferred

    obj1, obj2 = cfg.obj1, cfg.obj2
    is_ps_arr = np.asarray(is_ps).astype(bool)

    tr_idx, te_idx = similarity_cluster_holdout(
        features_df, is_ps_arr, test_frac=test_frac, seed=seed,
    )
    if len(te_idx) < 2 or len(tr_idx) < 2:
        if log:
            log.warning(f"[moe heldout plot] split too small "
                        f"(train={len(tr_idx)} test={len(te_idx)}); skipping.")
        return {}

    scaler1, scaler2 = _fit_label_scalers(labels_df.iloc[tr_idx], [obj1, obj2], cfg.transform)
    bundle = _fit_experts_and_gate(
        features_df, labels_df, is_ps_arr, np.asarray(tr_idx),
        [obj1, obj2], cfg.transform, scaler1, scaler2, cfg.seed_base, cfg, log,
    )
    if bundle is None:
        if log:
            log.warning("[moe heldout plot] a regime too small in the train split; skipping.")
        return {}

    # Predict BOTH the 80% train and the 20% test with the same (train-fit)
    # bundle, so the panel shows the in-split fit and the generalization gap.
    def _frame(idx: np.ndarray) -> pd.DataFrame:
        return _build_fold_frame(
            bundle, features_df, labels_df,
            original_indices=list(range(len(features_df))),
            te_idx=np.asarray(idx), is_ps=is_ps_arr, fold_idx=0,
            label_columns=[obj1, obj2], aux_col=cfg.aux1_obj1,
            ps_threshold=cfg.moe_threshold, scaler1=scaler1, scaler2=scaler2,
            transform=cfg.transform,
        )

    tr_true, tr_zm, tr_zv = _policy_from_frame(cfg, _frame(tr_idx))
    te_true, te_zm, te_zv = _policy_from_frame(cfg, _frame(te_idx))
    series = [
        {"name": "train (80%)", "true_z": tr_true, "pred_zm": tr_zm, "pred_zv": tr_zv,
         "color": "steelblue", "alpha": 0.25, "size": 8},
        {"name": "test (20%)", "true_z": te_true, "pred_zm": te_zm, "pred_zv": te_zv,
         "color": "orange", "alpha": 0.6, "size": 16},
    ]
    return _parity_panels(cfg, series, [obj1, obj2], "HELDOUT", f"MoE {cfg.moe_policy}", log=log)


def plot_global_holdout_fit(
    cfg: ALConfig,
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    is_ps: np.ndarray,
    *,
    log=None,
    test_frac: float = 0.2,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Similarity-cluster-holdout parity plot for the GLOBAL multitask GPR.

    Same protocol as `plot_moe_holdout_fit` (regime-stratified similarity split,
    train-only scalers, train+test overlay, z-space R²+NLPD), but with a single
    multitask GP fit on the 80% train (both regimes) instead of the mixture — so
    global vs MoE held-out numbers are directly comparable (identical split for a
    given seed). Writes `GPR_multitask_iter{N}_{tag}_HELDOUT_{label}.png`.
    """
    import copy
    from al_pipeline.diagnostic.al_regime_oof import _true_labels_to_z  # deferred
    from al_pipeline.surrogates.gpr_expert import GPRExpert             # deferred
    from al_pipeline.training.moe_training import FEATURE_COLUMNS, _fit_label_scalers

    obj1, obj2 = cfg.obj1, cfg.obj2
    is_ps_arr = np.asarray(is_ps).astype(bool)

    tr_idx, te_idx = similarity_cluster_holdout(
        features_df, is_ps_arr, test_frac=test_frac, seed=seed,
    )
    if len(te_idx) < 2 or len(tr_idx) < 2:
        if log:
            log.warning(f"[GPR multitask heldout] split too small "
                        f"(train={len(tr_idx)} test={len(te_idx)}); skipping.")
        return {}

    scaler1, scaler2 = _fit_label_scalers(labels_df.iloc[tr_idx], [obj1, obj2], cfg.transform)
    expert = GPRExpert.train(
        features_df.iloc[tr_idx].reset_index(drop=True),
        labels_df.iloc[tr_idx].reset_index(drop=True),
        [obj1, obj2], cfg.transform,
        copy.deepcopy(scaler1), copy.deepcopy(scaler2),
        FEATURE_COLUMNS, lr=cfg.learning_rate, epochs=cfg.epochs, patience=cfg.patience,
    )

    def _series(idx, name, color, alpha, size):
        feats = features_df.iloc[idx].reset_index(drop=True)
        labs = labels_df.iloc[idx].reset_index(drop=True)
        out = expert.predict(feats)
        zm = np.column_stack([out[f"{obj1}_z_mean"], out[f"{obj2}_z_mean"]])
        zv = np.column_stack([out[f"{obj1}_z_var"], out[f"{obj2}_z_var"]])
        tz = _true_labels_to_z(labs, obj1, obj2, cfg.transform, scaler1, scaler2)
        return {"name": name, "true_z": tz, "pred_zm": zm, "pred_zv": zv,
                "color": color, "alpha": alpha, "size": size}

    series = [
        _series(tr_idx, "train (80%)", "steelblue", 0.25, 8),
        _series(te_idx, "test (20%)", "orange", 0.6, 16),
    ]
    return _parity_panels(cfg, series, [obj1, obj2], "HELDOUT", "GPR multitask", log=log)
