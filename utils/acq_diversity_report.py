"""Aggregate acquisition-sweep batch-diversity across methods.

Each ``--acq_test`` job writes ``batch_features_norm_{tag}.csv`` (the normalized
selected-batch matrix) and a feature-space ``batch_diversity_{tag}.csv`` under
``runs/ACQ_SWEEP/<MODEL>/DIAGNOSTIC/``. This script collects those, fits ONE
shared UMAP per model so every method's batch lives in the same 2-D embedding,
and produces:

1. A tidy summary CSV with, per (model, tag, block):
     - feature-space Vendi + mean cosine distance,
     - UMAP-space   Vendi + mean Euclidean distance,
     - a random-batch baseline (feature space) drawn from the pooled candidates.
2. A conservation table: across methods, the Spearman rank correlation between
   the feature-space and UMAP-space Vendi (and mean distance). rho ~ 1 is the
   evidence that the UMAP scatter is a faithful visual of batch diversity — the
   diversity ordering of methods survives the 29-D -> 2-D projection.
3. A scatter plot of feature-space vs UMAP-space Vendi per method.

Usage
-----
    python utils/acq_diversity_report.py \
        --acq_root runs/ACQ_SWEEP \
        --out_dir  runs/ACQ_SWEEP/_diversity_report

Rationale for the two kernels: cosine is scale-invariant and appropriate in the
29-D feature space; cosine is meaningless on 2-D UMAP coordinates (origin-
dependent), so the UMAP space uses an RBF kernel on Euclidean distance with the
median-heuristic bandwidth. The claim we validate is rank agreement, not equality
of the absolute scores.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from al_pipeline.diagnostic.batch_diversity import (
    FEATURE_BLOCKS,
    diversity_rows,
    diversity_report,
)

_TAG_RE = re.compile(r"batch_features_norm_(?P<tag>.+)\.csv$")


def find_batch_files(acq_root: Path) -> List[Dict]:
    """Discover persisted batch matrices under ``<acq_root>/<MODEL>/DIAGNOSTIC/``.

    Returns a list of ``{model, tag, path}`` dicts.
    """
    found: List[Dict] = []
    for path in sorted(acq_root.glob("*/DIAGNOSTIC/batch_features_norm_*.csv")):
        m = _TAG_RE.search(path.name)
        if not m:
            continue
        model = path.parent.parent.name
        found.append({"model": model, "tag": m.group("tag"), "path": path})
    return found


def _fit_shared_umap(background: np.ndarray, seed: int = 42):
    """Fit a single UMAP on a common background matrix.

    n_neighbors is capped below the sample count so small backgrounds still fit.
    """
    from umap import UMAP  # lazy: heavy import, cluster/env only

    n = background.shape[0]
    n_neighbors = int(max(2, min(15, n - 1)))
    reducer = UMAP(n_neighbors=n_neighbors, random_state=seed)
    reducer.fit(background)
    return reducer


def _random_baseline_report(
    pool: np.ndarray, n: int, n_draws: int = 50, seed: int = 0
) -> Dict[str, float]:
    """Mean feature-space diversity of random size-``n`` draws from ``pool``."""
    rng = np.random.default_rng(seed)
    n = min(n, pool.shape[0])
    vendis, dists, norms = [], [], []
    for _ in range(n_draws):
        idx = rng.choice(pool.shape[0], size=n, replace=False)
        rep = diversity_report(pool[idx], kernel="cosine", metric="cosine")
        vendis.append(rep["vendi"])
        norms.append(rep["vendi_norm"])
        dists.append(rep["mean_distance"])
    return {
        "n": n,
        "vendi": float(np.mean(vendis)),
        "vendi_norm": float(np.mean(norms)),
        "mean_distance": float(np.mean(dists)),
    }


def build_summary(acq_root: Path, seed: int = 42) -> pd.DataFrame:
    """Assemble the per-(model, tag, space, block) diversity summary."""
    files = find_batch_files(acq_root)
    if not files:
        raise FileNotFoundError(
            f"No batch_features_norm_*.csv under {acq_root}/*/DIAGNOSTIC/"
        )

    # Group by model so each force field gets its own UMAP + baseline pool.
    by_model: Dict[str, List[Dict]] = {}
    for f in files:
        by_model.setdefault(f["model"], []).append(f)

    all_rows: List[Dict] = []
    for model, entries in by_model.items():
        batches = {e["tag"]: pd.read_csv(e["path"]).values.astype(np.float64) for e in entries}
        pool = np.vstack(list(batches.values()))  # per-model candidate pool proxy
        reducer = _fit_shared_umap(pool, seed=seed)

        for tag, X in batches.items():
            # Feature space (cosine), per block.
            for r in diversity_rows(X, space="feature", kernel="cosine", metric="cosine"):
                all_rows.append({"model": model, "tag": tag, **r})
            # UMAP space (RBF kernel / Euclidean), overall only.
            emb = reducer.transform(X)
            for r in diversity_rows(
                emb, space="umap", kernel="rbf", metric="euclidean",
                blocks={"overall": slice(None)},
            ):
                all_rows.append({"model": model, "tag": tag, **r})
            # Random-batch baseline (feature space, overall).
            base = _random_baseline_report(pool, n=X.shape[0], seed=seed)
            all_rows.append({"model": model, "tag": tag, "space": "feature_random_baseline",
                             "block": "overall", **base})

    return pd.DataFrame(all_rows)


def conservation_spearman(summary: pd.DataFrame) -> pd.DataFrame:
    """Spearman rank correlation feature-space vs UMAP-space, across methods.

    Computed per model on the 'overall' block. Needs >= 3 methods to be
    meaningful; fewer returns NaN with a note.
    """
    from scipy.stats import spearmanr

    rows: List[Dict] = []
    ov = summary[summary["block"] == "overall"]
    for model, g in ov.groupby("model"):
        feat = g[g["space"] == "feature"].set_index("tag")
        umap = g[g["space"] == "umap"].set_index("tag")
        tags = sorted(set(feat.index) & set(umap.index))
        n = len(tags)
        entry = {"model": model, "n_methods": n}
        if n >= 3:
            for col, name in [("vendi", "vendi"), ("mean_distance", "mean_distance")]:
                rho, p = spearmanr(feat.loc[tags, col].values, umap.loc[tags, col].values)
                entry[f"spearman_{name}"] = float(rho)
                entry[f"pvalue_{name}"] = float(p)
        else:
            entry["note"] = "need >=3 methods for a meaningful rank correlation"
        rows.append(entry)
    return pd.DataFrame(rows)


def plot_feature_vs_umap(summary: pd.DataFrame, out_path: Path) -> None:
    """Scatter feature-space vs UMAP-space Vendi, one point per method."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ov = summary[summary["block"] == "overall"]
    feat = ov[ov["space"] == "feature"]
    umap = ov[ov["space"] == "umap"]
    merged = feat.merge(umap, on=["model", "tag"], suffixes=("_feat", "_umap"))
    if merged.empty:
        return

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for model, g in merged.groupby("model"):
        ax.scatter(g["vendi_feat"], g["vendi_umap"], s=60, label=model)
        for _, row in g.iterrows():
            ax.annotate(row["tag"], (row["vendi_feat"], row["vendi_umap"]),
                        fontsize=6, alpha=0.7)
    ax.set_xlabel("Vendi score — feature space (cosine)")
    ax.set_ylabel("Vendi score — UMAP space (RBF)")
    ax.set_title("Batch diversity conservation: feature vs UMAP")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acq_root", type=Path, default=Path("runs/ACQ_SWEEP"),
                    help="Root holding <MODEL>/DIAGNOSTIC/batch_features_norm_*.csv")
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="Output dir (default: <acq_root>/_diversity_report)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = args.out_dir or (args.acq_root / "_diversity_report")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(args.acq_root, seed=args.seed)
    summary.to_csv(out_dir / "batch_diversity_summary.csv", index=False)

    cons = conservation_spearman(summary)
    cons.to_csv(out_dir / "batch_diversity_conservation.csv", index=False)

    plot_feature_vs_umap(summary, out_dir / "feature_vs_umap_vendi.png")

    print(f"Wrote diversity report to {out_dir}")
    print(cons.to_string(index=False))


if __name__ == "__main__":
    main()
