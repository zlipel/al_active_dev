"""Compare feature-space batch diversity with and without pessimism.

This is a presentation aid, not a full benchmark. It reads the acquisition-
diagnostic cells written by submit/acq_diag_cell.sh (one per model, iter, seed,
method) and answers one question in a form an audience can read:

    How much more diverse is the proposed batch when we add pessimism to
    kriging-believer, and does that help grow in later rounds?

Diversity is the cosine-kernel Vendi score on normalized feature vectors.
It is an effective-rank measure of those features, not a literal count of
biologically distinct designs. Outcome improvement is not tested here.

Outputs (under <home_root>/<exp_name>/_report/):
  - diversity_by_method.csv   mean +/- sd Vendi per (round, method)
  - diversity_increase.csv    Vendi increase of each method over the kb baseline
  - diversity_increase.png    grouped bar chart, one bar per method per round

Usage:
    python utils/acq_diag_report.py --exp_name seminar_20260923 --model MPIPI
"""

from __future__ import annotations

import argparse
import os
import re
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from al_pipeline.diagnostic.batch_diversity import diversity_report

_CELL_RE = re.compile(
    r"^(?P<model>.+)_iter(?P<iter>\d+)_seed(?P<seed>\d+)_(?P<method>kb|pkb|independent)"
    r"_(?P<front>upper|lower)_(?P<budget>full|pilot)$"
)

# Fixed presentation order and labels. Order is the escalation of the method:
# no within-batch update -> kriging-believer -> kriging-believer + pessimism.
_METHOD_ORDER = ["independent", "kb", "pkb"]
_METHOD_LABEL = {
    "independent": "Independent\n(no KB)",
    "kb": "Kriging-believer",
    "pkb": "KB + pessimism",
}
# Okabe-Ito, colorblind-safe: neutral gray for the control, blue for KB, orange
# to distinguish KB + pessimism from its controls.
_METHOD_COLOR = {"independent": "#999999", "kb": "#0072B2", "pkb": "#E69F00"}


def _round_label(it: int) -> str:
    return f"Round {it}→{it + 1}"


def _cell_vendi(home: Path, model: str, complete: dict) -> float:
    """Read and validate the requested completed batch, not an arbitrary glob."""
    diag = home / model / "DIAGNOSTIC"
    tag = complete["tag"]
    X = pd.read_csv(diag / f"batch_features_norm_{tag}.csv").to_numpy(dtype=float)
    if X.shape != (complete["ngen"], 29) or not np.isfinite(X).all():
        raise ValueError(f"Incomplete/non-finite completed batch: {home}")
    d = pd.read_csv(diag / f"batch_diversity_{tag}.csv")
    overall = d[d["block"] == "overall"]
    if len(overall) != 1 or int(overall["n"].iloc[0]) != len(X):
        raise ValueError(f"Invalid overall diversity row: {home}")
    score = float(overall["vendi"].iloc[0])
    if not np.isclose(score, diversity_report(X)["vendi"]):
        raise ValueError(f"Persisted diversity disagrees with selected features: {home}")
    return score


def build_per_cell(home_root: Path, exp_name: str, model: Optional[str],
                   front: str = "upper", budget: str = "full") -> pd.DataFrame:
    exp_dir = home_root / exp_name
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"No experiment dir at {exp_dir}")
    rows: List[Dict] = []
    for child in sorted(exp_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        m = _CELL_RE.match(child.name)
        if not m or (model and m["model"] != model):
            continue
        if m["front"] != front or m["budget"] != budget:
            continue
        marker = child / "completed.json"
        if not marker.is_file():
            warnings.warn(f"Skipping incomplete cell: {child.name}")
            continue
        complete = json.loads(marker.read_text())
        if (complete["model"], complete["iteration"], complete["ga_seed"], complete["front"]) != (
            m["model"], int(m["iter"]), int(m["seed"]), front
        ):
            raise ValueError(f"Completion metadata disagrees with cell name: {child}")
        vendi = _cell_vendi(child, m["model"], complete)
        rows.append({
            "model": m["model"], "iter": int(m["iter"]), "seed": int(m["seed"]),
            "method": m["method"], "vendi": vendi, "front": front, "budget": budget,
            "batch_n": complete["ngen"], "ncands": complete["ncands"],
            "ga_max_iter": complete["ga_max_iter"],
            "base_fit": json.dumps(complete["base_fit"], sort_keys=True),
        })
    if not rows:
        raise FileNotFoundError(f"No cells with diversity found under {exp_dir}")
    return pd.DataFrame(rows).sort_values(["model", "iter", "method", "seed"])


def build_summary(per_cell: pd.DataFrame) -> pd.DataFrame:
    g = per_cell.groupby(["model", "front", "budget", "batch_n", "iter", "method"])["vendi"]
    out = g.agg(vendi_mean="mean", vendi_sd="std", n_seeds="count").reset_index()
    out["round"] = out["iter"].map(_round_label)
    return out


def build_increase(per_cell: pd.DataFrame) -> pd.DataFrame:
    """Within-seed contrasts against KB, rejecting mismatched starting fits."""
    rows: List[Dict] = []
    for (model, front, budget, it), g in per_cell.groupby(["model", "front", "budget", "iter"]):
        base = g[g["method"] == "kb"]
        if base.empty:
            continue
        for method in sorted(set(g["method"]) - {"kb"}):
            test = g[g["method"] == method]
            paired = base.merge(test, on="seed", suffixes=("_kb", "_test"), validate="one_to_one")
            if len(paired) < max(len(base), len(test)):
                warnings.warn(f"Unpaired seeds excluded: {model}, iteration {it}, {method}")
            if paired.empty:
                continue
            for key in ("base_fit", "batch_n", "ncands", "ga_max_iter"):
                if not (paired[f"{key}_kb"] == paired[f"{key}_test"]).all():
                    raise ValueError(f"Matched comparison differs in {key}: {model}, {it}, {method}")
            difference = paired["vendi_test"] - paired["vendi_kb"]
            percent = 100.0 * difference / paired["vendi_kb"]
            rows.append({
                "model": model, "iter": it, "round": _round_label(it),
                "front": front, "budget": budget, "method": method,
                "vendi_mean": paired["vendi_test"].mean(), "kb_baseline": paired["vendi_kb"].mean(),
                "increase_abs": difference.mean(), "increase_sd": difference.std(),
                "increase_pct": percent.mean(), "n_pairs": len(paired),
            })
    return pd.DataFrame(rows)


def plot_diversity(summary: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "Arial", "font.size": 13, "axes.titlesize": 15,
        "axes.labelsize": 13, "axes.spines.top": False, "axes.spines.right": False,
    })

    iters = sorted(summary["iter"].unique())
    methods = [m for m in _METHOD_ORDER if m in set(summary["method"])]
    n_methods = len(methods)
    x = np.arange(len(iters))
    width = 0.8 / max(n_methods, 1)
    n_seeds = int(summary["n_seeds"].max())

    fig, ax = plt.subplots(figsize=(max(5.0, 1.6 * len(iters) + 2.5), 5.0))
    for j, method in enumerate(methods):
        means, sds = [], []
        for it in iters:
            row = summary[(summary["iter"] == it) & (summary["method"] == method)]
            means.append(float(row["vendi_mean"].iloc[0]) if not row.empty else np.nan)
            sd = float(row["vendi_sd"].iloc[0]) if not row.empty else np.nan
            sds.append(0.0 if np.isnan(sd) else sd)
        offset = (j - (n_methods - 1) / 2) * width
        bars = ax.bar(x + offset, means, width, yerr=sds, capsize=4,
                      color=_METHOD_COLOR[method], label=_METHOD_LABEL[method],
                      edgecolor="white", linewidth=0.8)
        for rect, mean in zip(bars, means):
            if not np.isnan(mean):
                ax.annotate(f"{mean:.1f}", (rect.get_x() + rect.get_width() / 2,
                            rect.get_height()), textcoords="offset points",
                            xytext=(0, 3), ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([_round_label(it) for it in iters])
    batch_sizes = sorted(summary["batch_n"].unique())
    if len(batch_sizes) != 1 or len(summary["model"].unique()) != 1:
        raise ValueError("Plot one model and one batch size at a time")
    ax.set_ylabel(f"Feature-space diversity\n(Vendi score; batch of {batch_sizes[0]})")
    ax.set_title("Batch diversity with and without pessimism")
    counts = sorted(summary["n_seeds"].unique())
    subtitle = (f"error bars = SD; {counts[0]} seed(s) per bar" if len(counts) == 1
                else f"error bars = SD; {counts[0]}–{counts[-1]} seeds per bar")
    ax.text(0.5, 1.015, subtitle, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=10, color="#666666")
    ax.legend(frameon=False, loc="upper left")
    upper = (summary["vendi_mean"] + summary["vendi_sd"].fillna(0)).max()
    ax.set_ylim(0, max(1.0, upper * 1.2))
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _print_takeaway(increase: pd.DataFrame) -> None:
    if increase.empty:
        return
    pkb = increase[increase["method"] == "pkb"].sort_values("iter")
    if pkb.empty:
        return
    print("\nTakeaway:")
    for _, r in pkb.iterrows():
        print(f"  {r['round']}: feature-space Vendi changes from "
              f"{r['kb_baseline']:.1f} (KB) to {r['vendi_mean']:.1f} "
              f"(KB+pessimism), {r['increase_pct']:+.0f}% over {r['n_pairs']} paired seed(s).")
    print("  These are proposal-feature contrasts, not measured property improvements.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp_name", required=True)
    ap.add_argument("--model", default=None, help="Restrict to one model (e.g. MPIPI)")
    ap.add_argument("--front", choices=["upper", "lower"], default="upper")
    ap.add_argument("--budget", choices=["full", "pilot"], default="full")
    ap.add_argument("--home_root", type=Path,
                    default=Path(os.environ.get("HOME_AL_ACQ",
                                 str(Path.home() / "PROJECTS/al_active_dev/runs/ACQ_SWEEP"))))
    ap.add_argument("--out_dir", type=Path, default=None)
    args = ap.parse_args()
    if not args.home_root.is_absolute():
        raise SystemExit("--home_root must be an absolute path")

    per_cell = build_per_cell(args.home_root, args.exp_name, args.model, args.front, args.budget)
    summary = build_summary(per_cell)
    increase = build_increase(per_cell)

    out_dir = args.out_dir or (args.home_root / args.exp_name / "_report")
    out_dir.mkdir(parents=True, exist_ok=True)
    per_cell.to_csv(out_dir / "diversity_per_cell.csv", index=False)
    summary.to_csv(out_dir / "diversity_by_method.csv", index=False)
    increase.to_csv(out_dir / "diversity_increase.csv", index=False)
    for model, subset in summary.groupby("model"):
        filename = "diversity_increase.png" if len(summary["model"].unique()) == 1 else f"diversity_{model}.png"
        plot_diversity(subset, out_dir / filename)

    pd.set_option("display.width", 140)
    print("=== feature-space diversity by method ===")
    print(summary[["model", "round", "method", "vendi_mean", "vendi_sd", "n_seeds"]]
          .to_string(index=False))
    if not increase.empty:
        print("\n=== increase over kriging-believer baseline ===")
        print(increase[["model", "round", "method", "vendi_mean", "kb_baseline",
                        "increase_abs", "increase_pct"]].to_string(index=False))
    _print_takeaway(increase)
    print(f"\nFigure + CSVs written to {out_dir}")


if __name__ == "__main__":
    main()
