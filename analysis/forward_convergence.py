"""Forward-convergence analysis: per (model, front) EHVI decay, proposed batch vs
current front, and predicted HV gain from submit/forward_round.sh outputs. Writes
a summary CSV + two plots; no simulation. See forward_convergence_results.md for
the recipe, --help for flags."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from al_pipeline.core.config import ALConfig
from al_pipeline.data_prep.parents import find_pareto_front
from al_pipeline.diagnostic._common import compute_hv_raw, _global_ref_point

# Deck palette (matches MOE_SUMMARY): global=blue, soft=orange, hard=green.
MODE_COLORS = {"global": "tab:blue", "soft": "tab:orange", "hard": "tab:green"}


def make_cfg(model: str, front: str, mode: str, iteration: int, *,
             fwd_root: Path, db_path: Path,
             ehvi_variant: str, exploration_strategy: str, transform: str,
             obj1: str, obj2: str) -> ALConfig:
    """Rebuild the exact cfg a forward_round.sh run used, for path + surrogate resolution."""
    train_model_type = "gpr_multitask" if mode == "global" else "moe"
    moe_policy = "hard" if mode == "hard" else "soft"  # ignored for global
    return ALConfig(
        model=model, iteration=iteration, front=front,
        train_model_type=train_model_type, moe_policy=moe_policy,
        run_tag=mode,
        ehvi_variant=ehvi_variant, exploration_strategy=exploration_strategy,
        transform=transform, obj1=obj1, obj2=obj2,
        base_path=fwd_root / "home", scratch_path=fwd_root / "scratch",
        db_path=db_path, pessimism=True, skip_data_prep=True,
    )


def read_ehvi(cfg: ALConfig) -> pd.DataFrame | None:
    path = cfg.paths.ga_children_dir / f"ehvi_values_{cfg.paths.tag}.csv"
    if not path.exists():
        print(f"  [warn] missing EHVI file: {path}")
        return None
    return pd.read_csv(path).sort_values("Seq_ID").reset_index(drop=True)


def read_batch_seqs(cfg: ALConfig) -> list[str]:
    path = cfg.paths.next_iter_candidates_file
    if not path.exists():
        print(f"  [warn] missing candidates file: {path}")
        return []
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def predict_batch_phys(cfg: ALConfig, seqs: list[str], db_path: Path) -> pd.DataFrame | None:
    """Physical (obj1, obj2) point estimate for the proposed batch via predict_design."""
    if not seqs:
        return None
    try:
        from al_pipeline.surrogates.loader import load_surrogate
        from al_pipeline.featurization.sequence_featurizer import SequenceFeaturizer
        surrogate = load_surrogate(cfg)
        X = SequenceFeaturizer(cfg.model.lower(), str(db_path)).featurize_many(seqs)
        pred = surrogate.predict_design(X)
    except Exception as exc:  # missing checkpoints, DB, etc. — degrade gracefully
        print(f"  [warn] could not predict batch phys ({cfg.run_tag}): {exc}")
        return None
    if pred.phys_mean is None:
        print(f"  [warn] surrogate returned no phys_mean ({cfg.run_tag})")
        return None
    return pd.DataFrame(np.asarray(pred.phys_mean), columns=[cfg.obj1, cfg.obj2]).dropna()


def load_front_labels(fwd_root: Path, model: str, iteration: int,
                      obj1: str, obj2: str) -> pd.DataFrame | None:
    """Current empirical labels (the curated seed) that define the front + ref point."""
    path = fwd_root / "scratch" / model / "GENERATIONS" / f"iteration_{iteration}" / f"labels_gen{iteration}.csv"
    if not path.exists():
        print(f"  [warn] missing labels for front: {path}")
        return None
    df = pd.read_csv(path)
    return df[[obj1, obj2]].dropna().reset_index(drop=True)


def n_batch_on_combined_front(labels: pd.DataFrame, batch: pd.DataFrame,
                              front: str, obj1: str, obj2: str) -> int:
    """How many proposed points sit on the front of (labels ∪ batch) — i.e. actually
    (weakly) advance it. Zero ⇒ every proposed point is dominated by current data."""
    kind = ["max", "max"] if front == "upper" else ["min", "min"]
    n_labels = len(labels)
    combined = pd.concat([labels[[obj1, obj2]], batch[[obj1, obj2]]], ignore_index=True)
    _, nd_idx = find_pareto_front(combined, kind=kind, objectives=[obj1, obj2])
    return int(sum(1 for i in nd_idx if i >= n_labels))


def analyze_cell(model, front, iteration, modes, *, fwd_root, db_path,
                 ehvi_variant, exploration_strategy, transform, obj1, obj2, out_dir):
    print(f"[{model} / {front}]")
    labels = load_front_labels(fwd_root, model, iteration, obj1, obj2)
    ref = hv_cur = None
    front_pts = None
    if labels is not None:
        ref = _global_ref_point(labels, front, obj1, obj2)
        hv_cur = compute_hv_raw(labels, front, obj1, obj2, ref)
        kind = ["max", "max"] if front == "upper" else ["min", "min"]
        front_pts, _ = find_pareto_front(labels, kind=kind, objectives=[obj1, obj2])

    rows, ehvi_by_mode, batch_by_mode = [], {}, {}
    for mode in modes:
        cfg = make_cfg(model, front, mode, iteration, fwd_root=fwd_root, db_path=db_path,
                       ehvi_variant=ehvi_variant, exploration_strategy=exploration_strategy,
                       transform=transform, obj1=obj1, obj2=obj2)
        ehvi = read_ehvi(cfg)
        seqs = read_batch_seqs(cfg)
        batch = predict_batch_phys(cfg, seqs, db_path) if labels is not None else None
        ehvi_by_mode[mode] = ehvi
        batch_by_mode[mode] = batch

        row = {"model": model, "front": front, "mode": mode,
               "n_picks": 0 if ehvi is None else len(ehvi),
               "ehvi_pick1": np.nan, "ehvi_last": np.nan, "ehvi_max": np.nan,
               "n_batch": 0 if batch is None else len(batch),
               "n_batch_advancing": np.nan,
               "hv_current": hv_cur, "hv_with_batch": np.nan,
               "hv_gain": np.nan, "hv_gain_frac": np.nan}
        if ehvi is not None and len(ehvi):
            row["ehvi_pick1"] = float(ehvi["EHVI"].iloc[0])
            row["ehvi_last"] = float(ehvi["EHVI"].iloc[-1])
            row["ehvi_max"] = float(ehvi["EHVI"].max())
        if labels is not None and batch is not None and len(batch):
            combined = pd.concat([labels, batch], ignore_index=True)
            hv_batch = compute_hv_raw(combined, front, obj1, obj2, ref)
            row["hv_with_batch"] = hv_batch
            row["hv_gain"] = hv_batch - hv_cur
            row["hv_gain_frac"] = (hv_batch - hv_cur) / hv_cur if hv_cur else np.nan
            row["n_batch_advancing"] = n_batch_on_combined_front(labels, batch, front, obj1, obj2)
        rows.append(row)

    _plot_ehvi_decay(ehvi_by_mode, model, front, out_dir)
    if front_pts is not None:
        _plot_front_scatter(front_pts, batch_by_mode, model, front, obj1, obj2, out_dir)
    return rows


def _plot_ehvi_decay(ehvi_by_mode, model, front, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    for mode, ehvi in ehvi_by_mode.items():
        if ehvi is None or not len(ehvi):
            continue
        ax.plot(ehvi["Seq_ID"], ehvi["EHVI"], marker="o", label=mode,
                color=MODE_COLORS.get(mode))
        plotted = True
    ax.set_xlabel("kriging-believer pick (Seq_ID)")
    ax.set_ylabel("leading EHVI")
    ax.set_title(f"{model} / {front} — EHVI decay across the batch")
    ax.axhline(0.0, color="0.7", lw=0.8, ls="--")
    if plotted:
        ax.legend()
    fig.tight_layout()
    out = out_dir / f"ehvi_decay_{model}_{front}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def _plot_front_scatter(front_pts, batch_by_mode, model, front, obj1, obj2, out_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    fp = front_pts.sort_values(obj1)
    ax.plot(fp[obj1], fp[obj2], "-o", color="0.3", ms=4, lw=1,
            label="current empirical front")
    for mode, batch in batch_by_mode.items():
        if batch is None or not len(batch):
            continue
        ax.scatter(batch[obj1], batch[obj2], s=45, alpha=0.8,
                   color=MODE_COLORS.get(mode), label=f"{mode} proposed")
    ax.set_xlabel(obj1)
    ax.set_ylabel(obj2)
    ax.set_title(f"{model} / {front} — proposed batch vs current front")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = out_dir / f"front_scatter_{model}_{front}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fwd_root", type=Path, required=True,
                    help="isolated tree root (contains scratch/ and home/)")
    ap.add_argument("--db_path", type=Path, required=True, help="feature DB for the featurizer")
    ap.add_argument("--models", nargs="+", default=["CALVADOS", "HPS_URRY", "MPIPI"])
    ap.add_argument("--fronts", nargs="+", default=["upper", "lower"])
    ap.add_argument("--modes", nargs="+", default=["global", "soft", "hard"])
    ap.add_argument("--iter", dest="iteration", type=int, default=10)
    ap.add_argument("--ehvi_variant", default="epsilon")
    ap.add_argument("--exploration_strategy", default="kriging_believer")
    ap.add_argument("--transform", default="yeoj")
    ap.add_argument("--obj1", default="exp_density")
    ap.add_argument("--obj2", default="diff")
    ap.add_argument("--out_dir", type=Path, default=Path("analysis/_forward_convergence"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for model in args.models:
        for front in args.fronts:
            all_rows += analyze_cell(
                model, front, args.iteration, args.modes,
                fwd_root=args.fwd_root, db_path=args.db_path,
                ehvi_variant=args.ehvi_variant,
                exploration_strategy=args.exploration_strategy,
                transform=args.transform, obj1=args.obj1, obj2=args.obj2,
                out_dir=args.out_dir,
            )

    summary = pd.DataFrame(all_rows)
    out_csv = args.out_dir / "forward_convergence_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
