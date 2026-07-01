"""
CLI entry point for the AL MoE retrospective diagnostic.

**Path split (matches the rest of the pipeline):**
- **Reads** training data (features_gen*.csv, labels_gen*.csv, seq_gen*.txt)
  from `cfg.scratch_path/<MODEL>/GENERATIONS/` — the same place every other
  training / GA consumer reads from. Override via `--runs_root` if you want
  to point at a copy elsewhere.
- **Writes** diagnostic outputs to `cfg.base_path/<MODEL>/DIAGNOSTIC/`
  (home-side, small + persistent).

Example (single model, upper front):

    python -m al_pipeline.cli.moe_diagnostic \\
        --model MPIPI --iter 0 --front upper \\
        --train_model_type moe --ehvi_variant epsilon \\
        --exploration_strategy kriging_believer \\
        --transform yeoj --obj1 exp_density --obj2 diff \\
        --base_path   $HOME/PROJECTS/al_active_dev/runs \\
        --scratch_path /scratch/gpfs/zl4808/PROJECTS/MODEL_COMPARISON \\
        --n_iters 10

Cheap enough to run on a login node — no LAMMPS, small kfold, ~10 min.
Produces three artifacts under `<base_path>/<MODEL>/DIAGNOSTIC/`:

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
    # Optional override for the completed-run source root. Defaults to
    # cfg.scratch_path (i.e. where the AL loop actually writes features and
    # labels). Only set this if you've copied a completed run somewhere else
    # (e.g. an archived snapshot).
    parser.add_argument("--runs_root", type=Path, default=None,
                          help="Override for the completed-run source root "
                               "(should contain <MODEL>/GENERATIONS/iteration_*). "
                               "Defaults to cfg.scratch_path.")
    parser.add_argument("--n_iters", type=int, default=10,
                          help="Number of completed iters to walk (default: 10).")
    parser.add_argument("--k_pick", type=int, default=None,
                          help="Top-K children to pick per iter under each surrogate. "
                               "Defaults to cfg.ngen // 2 (a 'half budget' retrospective).")
    parser.add_argument("--pessimism_start_iter", type=int, default=6,
                          help="First iter at which pessimism kicks in inside the KB inner "
                               "loop (default: 6, matches production practice of "
                               "no-pessimism rounds 1-5, pessimism rounds 6+).")
    parser.add_argument("--start_iter", type=int, default=1,
                          help="First iter to evaluate. Iters 0..start_iter-1 are folded "
                               "into every policy's initial picks as real data, so "
                               "divergence begins at start_iter. Useful for sweeping the "
                               "MoE-vs-global comparison from later starting points where "
                               "the PS training set is large enough (default: 1).")
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

    # Default runs_root to cfg.scratch_path — that's where every other
    # pipeline consumer reads features/labels from, so the diagnostic
    # matches by default. Explicit --runs_root overrides.
    runs_root = diag_args.runs_root if diag_args.runs_root is not None else cfg_base.scratch_path
    log.info(f"reading completed-run artifacts from {runs_root}")

    out = run_retrospective(
        runs_root=runs_root,
        model=cfg_base.model,
        cfg_base=cfg_base,
        n_iters=diag_args.n_iters,
        k_pick=diag_args.k_pick,
        pessimism_start_iter=diag_args.pessimism_start_iter,
        start_iter=diag_args.start_iter,
        log=log,
    )

    # Suffix all outputs by start_iter so sweeps don't clobber each other.
    suffix = f"_start{diag_args.start_iter}"
    diag_dir = cfg_base.paths.diagnostic_dir
    plot_path = diag_dir / f"retrospective_hv{suffix}.png"
    _plot_hv(out["trajectory"], plot_path)
    log.info(f"wrote plot: {plot_path}")

    # Concise summary to stdout.
    r2 = out["trajectory"]["rounds_to_95pct"]
    print(json.dumps({
        "target_hv":            out["target_hv"],
        "rounds_to_95pct_hv":   r2,
        "start_iter":           diag_args.start_iter,
        "summary_csv":          str(diag_dir / f"retrospective_summary{suffix}.csv"),
        "trajectory_json":      str(diag_dir / f"retrospective_trajectory{suffix}.json"),
        "plot":                 str(plot_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
