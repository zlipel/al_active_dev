"""
CLI entry point for the AL MoE regime-OOF diagnostic + final training.

Sister to `moe_diagnostic` (retrospective) and `moe_forward_diagnostic`.
Runs stratified k-fold OOF on ONE iteration's labeled data and (optionally)
trains the final production models the beam search will consume.

Reads training data from `cfg.paths.features_csv` + `cfg.paths.labels_csv`
(iteration N); writes outputs under `cfg.paths.diagnostic_dir` + saves
final models via the existing production training paths.

Example:

    python -m al_pipeline.cli.moe_regime_oof \\
        --model HPS_URRY --iter 10 --front upper \\
        --train_model_type moe --ehvi_variant epsilon \\
        --exploration_strategy kriging_believer \\
        --transform yeoj --obj1 exp_density --obj2 diff \\
        --base_path   $HOME/PROJECTS/al_active_dev/runs \\
        --scratch_path /scratch/gpfs/zl4808/PROJECTS/MODEL_COMPARISON \\
        --n_folds 5

Produces four CSVs + one JSON + eight PNGs under
`<base_path>/<MODEL>/DIAGNOSTIC/`:

  - regime_oof_predictions_iter{N}.csv
  - regime_oof_metrics_iter{N}.csv
  - regime_oof_summary_iter{N}.csv
  - regime_oof_classifier_iter{N}.csv
  - regime_oof_metadata_iter{N}.json
  - regime_oof_{rmse,r2,nll_z,spearman}_{all,ps,nonps}_iter{N}.png
  - regime_oof_classifier_reliability_iter{N}.png
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from al_pipeline.core.config import ALConfig
from al_pipeline.diagnostic.al_regime_oof import (
    HARD_THRESHOLDS, OOF_PREDICTORS, run_regime_oof,
)


def _parse_diagnostic_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Peel off regime-OOF args; hand the rest to ALConfig.from_cli."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Stratified k-fold splits for OOF (default: 5).")
    parser.add_argument("--ps_threshold", type=float, default=0.5,
                        help="Default gate threshold for moe_hard/ps_guarded "
                             "(threshold sweep is done at metrics time; default: 0.5).")
    parser.add_argument("--skip_oof", action="store_true",
                        help="Skip the OOF diagnostic (only train final models).")
    parser.add_argument("--skip_final", action="store_true",
                        help="Skip final production-model training (only run OOF).")
    args, remaining = parser.parse_known_args(argv)
    return args, remaining


# Match the forward diagnostic's colors so soft/hard curves are recognizable
# across the two diagnostic families. Global/experts get distinct colors.
_PREDICTOR_STYLE = {
    "global":       {"color": "tab:blue",   "linestyle": "-",  "linewidth": 1.6, "label": "global"},
    "ps_expert":    {"color": "tab:red",    "linestyle": "-",  "linewidth": 1.4, "label": "PS expert"},
    "nonps_expert": {"color": "tab:purple", "linestyle": "-",  "linewidth": 1.4, "label": "nonPS expert"},
    "moe_soft":     {"color": "tab:orange", "linestyle": "-",  "linewidth": 1.6, "label": "soft"},
    "moe_hard":     {"color": "#238b45",    "linestyle": "--", "linewidth": 1.4, "label": "hard@0.50"},
    "ps_guarded":   {"color": "tab:brown",  "linestyle": "--", "linewidth": 1.4, "label": "ps_guarded@0.50"},
}


_METRIC_SPECS: dict[str, dict] = {
    "rmse":     {"ylabel": "RMSE_z",     "hline": None},
    "r2":       {"ylabel": "R²",         "hline": (0.0, "grey", ":")},
    "nll_z":    {"ylabel": "NLL_z",      "hline": None},
    "spearman": {"ylabel": "Spearman ρ", "hline": (0.0, "grey", ":")},
}

_SPLIT_TITLES = {
    "all":   "all rows",
    "PS":    "PS rows only (density > 0)",
    "nonPS": "nonPS rows only (density = 0)",
}


def _plot_metric_by_predictor(
    metrics_df: pd.DataFrame,
    metric_col: str,
    split: str,
    obj1: str,
    obj2: str,
    default_threshold: float,
    out_path: Path,
) -> None:
    """
    Two-panel bar plot: one panel per objective. One bar per predictor
    (gated predictors take the DEFAULT threshold; the full threshold sweep
    lives in the metrics CSV). z-space, chosen split.
    """
    spec = _METRIC_SPECS[metric_col]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), dpi=200)
    for ax, prop in zip(axes, (obj1, obj2)):
        labels: list[str] = []
        values: list[float] = []
        colors: list[str] = []
        for predictor in OOF_PREDICTORS:
            cell = metrics_df[
                (metrics_df["space"] == "z")
                & (metrics_df["property"] == prop)
                & (metrics_df["split"] == split)
                & (metrics_df["predictor"] == predictor)
            ]
            if predictor in ("moe_hard", "ps_guarded"):
                cell = cell[cell["threshold"] == default_threshold]
            else:
                cell = cell[cell["threshold"].isna()]
            if cell.empty:
                continue
            val = float(cell.iloc[0][metric_col])
            if not np.isfinite(val):
                continue
            style = _PREDICTOR_STYLE[predictor]
            labels.append(style["label"])
            values.append(val)
            colors.append(style["color"])
        xs = np.arange(len(values))
        ax.bar(xs, values, color=colors)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        if spec["hline"] is not None:
            val, color, ls = spec["hline"]
            ax.axhline(val, color=color, linestyle=ls, linewidth=0.7)
        ax.set_title(f"{prop} (z-space, {_SPLIT_TITLES[split]})")
        ax.set_ylabel(spec["ylabel"])
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(
        f"Regime OOF — {metric_col} — {_SPLIT_TITLES[split]}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_classifier_reliability(clf_row: dict, out_path: Path) -> None:
    """
    Two-panel plot: confusion counts + summary metrics bar.

    The RF-gate summary in the CSV is single-row; visual complement is a
    small confusion matrix + a bar of {ROC-AUC, PR-AUC, Accuracy, F1,
    Brier} so you can eyeball whether the calibrated gate is doing its
    job.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), dpi=200)

    # Left: 2x2 confusion counts
    ax = axes[0]
    cm = np.array([[clf_row["tn"], clf_row["fp"]],
                   [clf_row["fn"], clf_row["tp"]]], dtype=float)
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                    color="black" if cm[i, j] < cm.max() / 2 else "white",
                    fontsize=11, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred nonPS", "pred PS"], fontsize=9)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["true nonPS", "true PS"], fontsize=9)
    ax.set_title(f"Confusion (τ={clf_row['threshold']:.2f})", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)

    # Right: summary metric bars
    ax = axes[1]
    keys = ["roc_auc", "pr_auc", "accuracy", "f1", "recall", "precision", "brier"]
    labels = ["ROC-AUC", "PR-AUC", "Acc.", "F1", "Recall", "Precision", "Brier"]
    vals = [float(clf_row.get(k, np.nan)) for k in keys]
    xs = np.arange(len(vals))
    ax.bar(xs, vals, color="tab:blue")
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0.0, 1.05)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.7)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title(f"Gate summary (n={clf_row['n']}, "
                 f"n_ps={clf_row['n_ps']}, mean p_ps|PS={clf_row['mean_p_ps_true_ps']:.2f}, "
                 f"mean p_ps|nonPS={clf_row['mean_p_ps_true_nonps']:.2f})",
                 fontsize=9)

    fig.suptitle("Regime OOF — RF gate reliability", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("moe_regime_oof")

    argv = argv if argv is not None else sys.argv[1:]
    diag_args, rest = _parse_diagnostic_args(argv)
    _orig_argv = sys.argv
    sys.argv = [sys.argv[0], *rest]
    try:
        cfg = ALConfig.from_cli()
    finally:
        sys.argv = _orig_argv

    if diag_args.skip_oof and diag_args.skip_final:
        log.warning("Both --skip_oof and --skip_final set; nothing to do.")
        return 0

    out = run_regime_oof(
        cfg,
        n_folds=diag_args.n_folds,
        ps_threshold=diag_args.ps_threshold,
        skip_oof=diag_args.skip_oof,
        skip_final=diag_args.skip_final,
        log=log,
    )

    plot_paths: dict[str, dict[str, str]] = {m: {} for m in _METRIC_SPECS}
    classifier_plot_path: str | None = None
    diag_dir = cfg.paths.diagnostic_dir
    iter_n = cfg.iteration
    if not diag_args.skip_oof and len(out["metrics_df"]) > 0:
        for metric_col in _METRIC_SPECS:
            for split in ("all", "PS", "nonPS"):
                split_tag = split.lower()
                path = diag_dir / f"regime_oof_{metric_col}_{split_tag}_iter{iter_n}.png"
                _plot_metric_by_predictor(
                    out["metrics_df"], metric_col, split,
                    cfg.obj1, cfg.obj2, diag_args.ps_threshold, path,
                )
                plot_paths[metric_col][split_tag] = str(path)
                log.info(f"wrote plot: {path}")

        if out["classifier"]:
            classifier_plot_path = str(
                diag_dir / f"regime_oof_classifier_reliability_iter{iter_n}.png"
            )
            _plot_classifier_reliability(out["classifier"], Path(classifier_plot_path))
            log.info(f"wrote plot: {classifier_plot_path}")

    print(json.dumps({
        "iter":            iter_n,
        "n_folds":         diag_args.n_folds,
        "ps_threshold":    diag_args.ps_threshold,
        "predictions_csv": str(out["paths"]["predictions"]),
        "metrics_csv":     str(out["paths"]["metrics"]),
        "summary_csv":     str(out["paths"]["summary"]),
        "classifier_csv":  str(out["paths"]["classifier"]),
        "metadata_json":   str(out["paths"]["metadata"]),
        "final_models":    out["paths"]["final"],
        "plots": {
            **plot_paths,
            "classifier_reliability": classifier_plot_path,
        },
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
