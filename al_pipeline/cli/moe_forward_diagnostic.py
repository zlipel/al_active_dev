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

Produces four CSVs and one plot under `<base_path>/<MODEL>/DIAGNOSTIC/`:

  - forward_predictions_start{N}.csv
  - forward_metrics_start{N}.csv
  - forward_classifier_start{N}.csv
  - forward_ranking_start{N}.csv
  - forward_rmse_start{N}.png
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
    "global":       {"color": "tab:blue",   "linestyle": "-",  "linewidth": 1.5},
    "moe_soft":     {"color": "tab:orange", "linestyle": "-",  "linewidth": 1.5},
    "moe_hard":     {"color": "tab:green",  "linestyle": "-",  "linewidth": 1.5},
    "ps_expert":    {"color": "tab:red",    "linestyle": "--", "linewidth": 1.3},
    "nonps_expert": {"color": "tab:purple", "linestyle": "--", "linewidth": 1.3},
    "ps_guarded":   {"color": "tab:brown",  "linestyle": ":",  "linewidth": 1.3},
}


def _plot_rmse_by_iter(metrics_df: pd.DataFrame, obj1: str, obj2: str,
                         out_path: Path) -> None:
    """Two-panel plot: z-space RMSE vs. iter, one line per predictor, per objective.

    Aggregated across both fronts (split='all', front_type='all'). Users who
    want per-front curves can pivot the metrics_long CSV — the `front_type`
    column carries {all, upper, lower}."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), dpi=200, sharex=True)
    for ax, prop in zip(axes, (obj1, obj2)):
        cell = metrics_df[
            (metrics_df["space"] == "z")
            & (metrics_df["split"] == "all")
            & (metrics_df["front_type"] == "all")
            & (metrics_df["property"] == prop)
        ]
        for predictor in ALL_PREDICTORS:
            rows = cell[cell["predictor"] == predictor].sort_values("heldout_iter")
            if rows.empty:
                continue
            ax.plot(rows["heldout_iter"], rows["rmse"],
                    label=predictor, **_PREDICTOR_STYLE[predictor])
        ax.set_title(f"{prop} (z-space, split=all, both fronts)")
        ax.set_xlabel("held-out iter")
        ax.set_ylabel("RMSE_z")
        ax.grid(True, alpha=0.3)
    axes[-1].legend(loc="upper right", fontsize=7, frameon=False)
    fig.suptitle("Forward diagnostic — aggregated across upper + lower", fontsize=10)
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
    plot_path = diag_dir / f"forward_rmse{suffix}.png"
    if len(out["metrics_df"]) > 0:
        _plot_rmse_by_iter(
            out["metrics_df"], cfg_base.obj1, cfg_base.obj2, plot_path,
        )
        log.info(f"wrote plot: {plot_path}")

    print(json.dumps({
        "start_iter":       diag_args.start_iter,
        "n_iters":          diag_args.n_iters,
        "predictions_csv":  str(out["paths"]["predictions"]),
        "metrics_csv":      str(out["paths"]["metrics"]),
        "classifier_csv":   str(out["paths"]["classifier"]),
        "ranking_csv":      str(out["paths"]["ranking"]),
        "plot":             str(plot_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
