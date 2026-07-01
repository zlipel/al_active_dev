"""
CLI entry point for the AL MoE retrospective diagnostic.

Example (single model, upper front):

    python -m al_pipeline.cli.moe_diagnostic \\
        --model MPIPI --iter 0 --front upper \\
        --train_model_type moe --ehvi_variant epsilon \\
        --exploration_strategy kriging_believer \\
        --transform yeoj --obj1 exp_density --obj2 diff \\
        --runs_root $HOME/PROJECTS/al_active_dev/runs \\
        --n_iters 10

Cheap enough to run on a login node — no LAMMPS, small kfold, ~10 min.
Produces three artifacts under `<runs_root>/<MODEL>/DIAGNOSTIC/`:

  - retrospective_summary.csv   — one row per iter (HV under each policy, hits)
  - retrospective_trajectory.json — full HV curves + target front + rounds-to-95%
  - retrospective_hv.png         — HV vs. iter plot with all four curves

The CLI reuses `ALConfig.from_cli` so every AL tuning parameter is respected.
Only three retrospective-specific args are added.
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

from al_pipeline.core.config import ALConfig
from al_pipeline.diagnostic.al_retrospective import run_retrospective


def _parse_diagnostic_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """
    Peel off the retrospective-specific args and hand the rest to ALConfig.from_cli.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runs_root", type=Path, required=True,
                          help="Root containing <MODEL>/GENERATIONS/iteration_* (completed run).")
    parser.add_argument("--n_iters", type=int, default=10,
                          help="Number of completed iters to walk (default: 10).")
    parser.add_argument("--k_pick", type=int, default=None,
                          help="Top-K children to pick per iter under each surrogate. "
                               "Defaults to cfg.ngen // 2 (a 'half budget' retrospective).")
    args, remaining = parser.parse_known_args(argv)
    return args, remaining


def _plot_hv(trajectory: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 3.5), dpi=200)
    iters = trajectory["iters"]
    for label, key, style in (
        ("actual (full batch)", "hv_actual",   {"color": "black", "linestyle": "--", "linewidth": 1.5}),
        ("global",              "hv_global",   {"color": "tab:blue",   "linewidth": 1.5}),
        ("MoE soft",            "hv_moe_soft", {"color": "tab:orange", "linewidth": 1.5}),
        ("MoE hard",            "hv_moe_hard", {"color": "tab:green",  "linewidth": 1.5}),
    ):
        ax.plot(iters, trajectory[key], label=label, **style)
    target = trajectory["target_hv"]
    ax.axhline(target,         color="grey", linestyle=":", linewidth=1.0, label="target HV")
    ax.axhline(0.95 * target,  color="grey", linestyle=":", linewidth=0.7, alpha=0.6, label="95% target")
    ax.set_xlabel("AL iteration")
    ax.set_ylabel("cumulative HV (raw objective space)")
    ax.set_title(f"MoE retrospective — k_pick={trajectory['k_pick']}, front={trajectory['front']}")
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("moe_diagnostic")

    argv = argv if argv is not None else sys.argv[1:]
    diag_args, rest = _parse_diagnostic_args(argv)
    # ALConfig.from_cli parses sys.argv by default; swap it out temporarily.
    _orig_argv = sys.argv
    sys.argv = [sys.argv[0], *rest]
    try:
        cfg_base = ALConfig.from_cli()
    finally:
        sys.argv = _orig_argv

    out = run_retrospective(
        runs_root=diag_args.runs_root,
        model=cfg_base.model,
        cfg_base=cfg_base,
        n_iters=diag_args.n_iters,
        k_pick=diag_args.k_pick,
        log=log,
    )

    # Plot
    plot_path = cfg_base.paths.diagnostic_dir / "retrospective_hv.png"
    _plot_hv(out["trajectory"], plot_path)
    log.info(f"wrote plot: {plot_path}")

    # Concise summary to stdout.
    r2 = out["trajectory"]["rounds_to_95pct"]
    print(json.dumps({
        "target_hv":            out["target_hv"],
        "rounds_to_95pct_hv":   r2,
        "summary_csv":          str(cfg_base.paths.diagnostic_dir / "retrospective_summary.csv"),
        "trajectory_json":      str(cfg_base.paths.diagnostic_dir / "retrospective_trajectory.json"),
        "plot":                 str(plot_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
