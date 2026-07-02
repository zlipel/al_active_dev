"""
CLI entry point for the AL MoE forward (generation-forward) diagnostic.

Sister to `moe_diagnostic`. Same iter-walk pattern (train on gens 0..N-1,
evaluate on gen N); reports predictive accuracy — RMSE, MAE, R², bias,
Spearman, z-space NLL — across six predictors × two objectives × two spaces
(z, physical) × three splits (all, PS, nonPS), plus RF-gate quality per iter.

Reads training data from `cfg.scratch_path/<MODEL>/GENERATIONS/`; writes
outputs to `cfg.base_path/<MODEL>/DIAGNOSTIC/` (same split as the
retrospective — override --runs_root only if you're pointing at a snapshot
copy elsewhere).

Example:

    python -m al_pipeline.cli.moe_forward_diagnostic \\
        --model MPIPI --iter 0 --front upper \\
        --train_model_type moe --ehvi_variant epsilon \\
        --exploration_strategy kriging_believer \\
        --transform yeoj --obj1 exp_density --obj2 diff \\
        --base_path   $HOME/PROJECTS/al_active_dev/runs \\
        --scratch_path /scratch/gpfs/zl4808/PROJECTS/MODEL_COMPARISON \\
        --n_iters 10 --start_iter 1

Produces four CSVs and 13 plots under `<base_path>/<MODEL>/DIAGNOSTIC/`:

  - forward_predictions_start{N}.csv
  - forward_metrics_start{N}.csv
  - forward_classifier_start{N}.csv
  - forward_ranking_start{N}.csv
  - forward_{rmse,r2,nll_z,spearman}_{all,ps,nonps}_start{N}.png  (12)
  - forward_classifier_start{N}.png  (RF-gate quality; 2x2 grid)
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
import pandas as pd

from al_pipeline.core.config import ALConfig
from al_pipeline.diagnostic.al_forward import ALL_PREDICTORS, run_forward


def _parse_diagnostic_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Peel off the forward-specific args; hand the rest to ALConfig.from_cli."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runs_root", type=Path, default=None,
                          help="Override for the completed-run source root "
                               "(should contain <MODEL>/GENERATIONS/iteration_*). "
                               "Defaults to cfg.scratch_path.")
    parser.add_argument("--n_iters", type=int, default=10,
                          help="Number of completed iters to walk (default: 10).")
    parser.add_argument("--start_iter", type=int, default=1,
                          help="First iter to evaluate (default: 1). Iters 0..start_iter-1 "
                               "are not scored; useful for sweeping the diagnostic from "
                               "later starting points where the PS training set is large.")
    args, remaining = parser.parse_known_args(argv)
    return args, remaining


_PREDICTOR_STYLE = {
    "global":         {"color": "tab:blue",   "linestyle": "-",  "linewidth": 1.6, "label": "global"},
    "moe_soft":       {"color": "tab:orange", "linestyle": "-",  "linewidth": 1.6, "label": "soft"},
    "moe_hard_t015":  {"color": "#c7e9c0",    "linestyle": "--", "linewidth": 1.4, "label": "hard@0.15"},
    "moe_hard_t030":  {"color": "#74c476",    "linestyle": "--", "linewidth": 1.4, "label": "hard@0.30"},
    "moe_hard_t050":  {"color": "#238b45",    "linestyle": "--", "linewidth": 1.4, "label": "hard@0.50"},
    "moe_hard_t070":  {"color": "#00441b",    "linestyle": "--", "linewidth": 1.4, "label": "hard@0.70"},
}


# y-axis label + optional reference-line (value, color, linestyle) per metric.
# Reference lines mark the intuitive "no-skill" or "no-bias" level for each
# metric so eyeballing a single curve against it is meaningful.
_METRIC_SPECS: dict[str, dict] = {
    "rmse":     {"ylabel": "RMSE_z",     "hline": None},
    "r2":       {"ylabel": "R²",    "hline": (0.0, "grey", ":")},
    "nll_z":    {"ylabel": "NLL_z",      "hline": None},
    "spearman": {"ylabel": "Spearman ρ", "hline": (0.0, "grey", ":")},
}

_SPLIT_TITLES = {
    "all":   "all rows",
    "ps":    "PS rows only (density > 0)",
    "nonps": "nonPS rows only (density <= 0)",
}


def _plot_metric_by_iter(metrics_df: pd.DataFrame, metric_col: str, split: str,
                          obj1: str, obj2: str, out_path: Path) -> None:
    """Two-panel plot: `metric_col` vs. iter, one curve per predictor, per objective.

    Filters to `space='z'`, `front_type='all'`, and the chosen `split`. Users
    who want per-front breakdowns can pivot the metrics_long CSV — the
    `front_type` column carries {all, upper, lower}.

    Parameters
    ----------
    metrics_df : DataFrame
        Long-format metrics table from `run_forward` (one row per
        predictor × property × space × split × front_type × iter).
    metric_col : str
        Column to plot on the y-axis. Must be a key of `_METRIC_SPECS`.
    split : str
        Which row subset to plot: `'all' | 'ps' | 'nonps'`.
    obj1, obj2 : str
        Objective column names — one subplot each.
    out_path : Path
        PNG destination.
    """
    spec = _METRIC_SPECS[metric_col]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), dpi=200, sharex=True)
    for ax, prop in zip(axes, (obj1, obj2)):
        cell = metrics_df[
            (metrics_df["space"] == "z")
            & (metrics_df["split"] == split)
            & (metrics_df["front_type"] == "all")
            & (metrics_df["property"] == prop)
        ]
        for predictor in ALL_PREDICTORS:
            rows = cell[cell["predictor"] == predictor].sort_values("heldout_iter")
            if rows.empty:
                continue
            ax.plot(rows["heldout_iter"], rows[metric_col],
                    **_PREDICTOR_STYLE[predictor])
        if spec["hline"] is not None:
            val, color, ls = spec["hline"]
            ax.axhline(val, color=color, linestyle=ls, linewidth=0.7)
        ax.set_title(f"{prop} (z-space, {_SPLIT_TITLES[split]})")
        ax.set_xlabel("held-out iter")
        ax.set_ylabel(spec["ylabel"])
        ax.grid(True, alpha=0.3)
    axes[-1].legend(loc="best", fontsize=7, frameon=False)
    fig.suptitle(
        f"Forward diagnostic — {metric_col} — {_SPLIT_TITLES[split]}, both fronts",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_classifier_metrics(classifier_df: pd.DataFrame, out_path: Path) -> None:
    """2x2 grid of RF-gate quality metrics vs. heldout_iter (front_type='all').

    Panels: ROC-AUC, F1, PS recall, nonPS FPR. A single black curve per panel
    (no predictor split — the gate is one classifier). Reference lines mark
    the "no-skill" baseline (ROC-AUC = 0.5) and the "no false positive" floor
    (nonPS FPR = 0).
    """
    cell = classifier_df[classifier_df["front_type"] == "all"].sort_values("heldout_iter")

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0), dpi=200, sharex=True)
    panels = [
        (axes[0, 0], "roc_auc",     "ROC-AUC",  (0.5, "grey", ":")),
        (axes[0, 1], "f1",          "F1 (PS)",  None),
        (axes[1, 0], "ps_recall",   "PS recall", None),
        (axes[1, 1], "nonps_fpr",   "nonPS FPR", (0.0, "grey", ":")),
    ]
    for ax, col, ylabel, hline in panels:
        ax.plot(cell["heldout_iter"], cell[col], color="black", linewidth=1.5)
        if hline is not None:
            val, color, ls = hline
            ax.axhline(val, color=color, linestyle=ls, linewidth=0.7)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    for ax in axes[1, :]:
        ax.set_xlabel("held-out iter")
    fig.suptitle("Forward diagnostic — RF gate quality (both fronts)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("moe_forward_diagnostic")

    argv = argv if argv is not None else sys.argv[1:]
    diag_args, rest = _parse_diagnostic_args(argv)
    _orig_argv = sys.argv
    sys.argv = [sys.argv[0], *rest]
    try:
        cfg_base = ALConfig.from_cli()
    finally:
        sys.argv = _orig_argv

    runs_root = diag_args.runs_root if diag_args.runs_root is not None else cfg_base.scratch_path
    log.info(f"reading completed-run artifacts from {runs_root}")

    # cfg.front is required by ALConfig.from_cli but the forward diagnostic
    # covers both fronts in one run (train once, evaluate against both halves
    # of every gen-N pool). Warn if the user thought otherwise.
    log.warning(
        "--front is required by ALConfig but is IGNORED by the forward "
        "diagnostic. One run covers both fronts; front is a per-row "
        "attribute in the outputs (front_type column)."
    )

    out = run_forward(
        runs_root=runs_root,
        model=cfg_base.model,
        cfg_base=cfg_base,
        n_iters=diag_args.n_iters,
        start_iter=diag_args.start_iter,
        log=log,
    )

    suffix = f"_start{diag_args.start_iter}"
    diag_dir = cfg_base.paths.diagnostic_dir
    plot_paths: dict[str, dict[str, str]] = {m: {} for m in _METRIC_SPECS}
    classifier_plot_path: str | None = None

    if len(out["metrics_df"]) > 0:
        for metric_col in _METRIC_SPECS:
            for split in ("all", "ps", "nonps"):
                path = diag_dir / f"forward_{metric_col}_{split}{suffix}.png"
                _plot_metric_by_iter(
                    out["metrics_df"], metric_col, split,
                    cfg_base.obj1, cfg_base.obj2, path,
                )
                plot_paths[metric_col][split] = str(path)
                log.info(f"wrote plot: {path}")

    if len(out["classifier_df"]) > 0:
        classifier_plot_path = str(diag_dir / f"forward_classifier{suffix}.png")
        _plot_classifier_metrics(out["classifier_df"], Path(classifier_plot_path))
        log.info(f"wrote plot: {classifier_plot_path}")

    print(json.dumps({
        "start_iter":       diag_args.start_iter,
        "n_iters":          diag_args.n_iters,
        "predictions_csv":  str(out["paths"]["predictions"]),
        "metrics_csv":      str(out["paths"]["metrics"]),
        "classifier_csv":   str(out["paths"]["classifier"]),
        "ranking_csv":      str(out["paths"]["ranking"]),
        "plots": {
            **plot_paths,
            "classifier": classifier_plot_path,
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
