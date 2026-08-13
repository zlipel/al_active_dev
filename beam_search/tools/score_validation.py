"""Score beam-search validation sims against surrogate predictions.

For each simulated endpoint, compare the beam surrogate's predicted physical
value to the actual simulation result and report how well they correlate.
The headline question is directional-agnostic: does the model's endpoint
prediction track the endpoint simulation (target Pearson r >= 0.8)?

Two axes, keyed off the ``exp_density`` (u / obj1) and ``diff`` (v / obj2)
observables. The diff axis auto-activates once ``diffusivities.csv`` exists;
until then it is reported as "no sim data yet" and skipped.

Inputs, per (model, axis), all under the validation SIMULATIONS tree written
by gen_validation_sequences.py + the analysis stage:

    <scratch>/<MODEL>/VALIDATION/<SCOPE>/SIMULATIONS/<PHASE>/
        ├── endpoints_metadata.csv   sim_row (join) + <pred_col> (prediction)
        └── <sim_file>               seq_ID (join) + <sim_col> (simulation)

Join key: metadata ``sim_row`` == sim ``seq_ID`` (both index the deduplicated
seq_<scope>.txt line order). ``end_seq`` is cross-checked against the sim
``sequence`` column.

Outputs land beside the beam endpoint predictions, under the mode/policy tree:

    <paths_root>/<MODEL>/<SCOPE>/<POLICY>/
        ├── validation_endpoints_<axis>.csv   per-endpoint pred vs sim + error
        ├── validation_summary_<axis>.csv      this model's corr breakdown
        └── validation_parity_<axis>.png       this model's parity plot
    <paths_root>/ALL/<SCOPE>/<POLICY>/          cross-model aggregate
        ├── validation_summary_<axis>.csv      full breakdown (all scopes)
        └── validation_parity_<axis>.png        combined parity plot

Example (local):

    python beam_search/tools/score_validation.py \\
        --scratch_dir runs \\
        --paths_root runs/beam_inspection/PATHS \\
        --scope BENCHMARK --policy expert_tied --length_changes

On the cluster --scratch_dir is $SCRATCH_AL and --paths_root defaults to
$SCRATCH_AL/PATHS (or PATHS_FIXED_LENGTH without --length_changes).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# axis -> (SIMULATIONS phase dir, sim CSV, sim value col, sim std col, metadata prediction col)
AXES = {
    "eos": dict(phase="EOS", sim_file="eos_results.csv",
                sim_col="exp_density", sim_std="exp_density_std", pred_col="rho_end"),
    "diff": dict(phase="DIFF", sim_file="diffusivities.csv",
                 sim_col="diff", sim_std="diff_std", pred_col="diff_end"),
}
REGIME_KEEP = 0.5   # p_ps split: model "calls PS" when endpoint_p_ps >= this


def load_axis(scratch_dir: Path, model: str, scope: str, axis: str) -> pd.DataFrame:
    """Join one model's simulated endpoints to their surrogate predictions.

    Returns an empty frame if the sim CSV for this axis does not exist yet
    (e.g. the diff campaign has not run).
    """
    cfg = AXES[axis]
    sim_dir = scratch_dir / model / "VALIDATION" / scope / "SIMULATIONS" / cfg["phase"]
    meta_path = sim_dir / "endpoints_metadata.csv"
    sim_path = sim_dir / cfg["sim_file"]
    if not meta_path.exists() or not sim_path.exists():
        return pd.DataFrame()

    meta = pd.read_csv(meta_path)
    sim = pd.read_csv(sim_path).set_index("seq_ID")
    rows = []
    for _, m in meta.iterrows():
        sid = int(m["sim_row"])
        if sid not in sim.index:
            continue
        s = sim.loc[sid]
        if str(s["sequence"]) != str(m["end_seq"]):
            raise ValueError(
                f"[{model}/{axis}] sim_row {sid} sequence mismatch: "
                f"metadata end_seq != {cfg['sim_file']} sequence"
            )
        row = dict(
            model=model,
            axis=axis,
            regime=m["start_regime"],
            category=m.get("category", ""),
            start_idx=int(m["start_idx"]),
            du_req=float(m["du_req"]),
            dv_req=float(m["dv_req"]),
            pred=float(m[cfg["pred_col"]]),
            sim=float(s[cfg["sim_col"]]),
            sim_std=float(s[cfg["sim_std"]]) if cfg["sim_std"] in s else np.nan,
            start_p_ps=float(m["start_p_ps"]),
            endpoint_p_ps=float(m["endpoint_p_ps"]),
            end_seq=str(m["end_seq"]),
        )
        if axis == "eos" and "psp" in s:
            row["sim_psp"] = int(s["psp"])
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["abs_err"] = (df["pred"] - df["sim"]).abs()
    return df


def _corr_row(scope: str, sub: pd.DataFrame, min_corr: float) -> dict:
    n = len(sub)
    ok_spread = n >= 3 and sub["sim"].nunique() >= 2 and sub["pred"].nunique() >= 2
    pr = pearsonr(sub["sim"], sub["pred"])[0] if ok_spread else np.nan
    sr = spearmanr(sub["sim"], sub["pred"])[0] if ok_spread else np.nan
    return dict(
        scope=scope, n=n,
        pearson=round(pr, 4) if ok_spread else np.nan,
        spearman=round(sr, 4) if ok_spread else np.nan,
        mae=round((sub["pred"] - sub["sim"]).abs().mean(), 4) if n else np.nan,
        bias=round((sub["pred"] - sub["sim"]).mean(), 4) if n else np.nan,
        passes=(bool(pr >= min_corr) if ok_spread else False),
    )


def breakdown(df: pd.DataFrame, min_corr: float, per_model: bool) -> pd.DataFrame:
    """Full correlation breakdown: overall, per-model, per-regime, per cell."""
    rows = [_corr_row("ALL", df, min_corr)]
    models = sorted(df["model"].unique())
    if per_model:
        for m in models:
            rows.append(_corr_row(m, df[df.model == m], min_corr))
    for reg in ["ps", "nonps"]:
        sub = df[df.regime == reg]
        if len(sub):
            rows.append(_corr_row(f"regime:{reg}", sub, min_corr))
    if per_model:
        for m in models:
            for reg in ["ps", "nonps"]:
                sub = df[(df.model == m) & (df.regime == reg)]
                if len(sub):
                    rows.append(_corr_row(f"{m}/{reg}", sub, min_corr))
    return pd.DataFrame(rows)


def parity_plot(df: pd.DataFrame, axis: str, title: str, out_png: Path) -> None:
    markers = {"MPIPI": "o", "HPS_URRY": "s", "CALVADOS": "^", "MARTINI": "D", "HPS_KR": "v"}
    colors = {"ps": "#c0392b", "nonps": "#2c7fb8"}
    # diff spans ~2 orders of magnitude (PS ~0.1 -> nonPS ~40); log scale keeps
    # the low-diff PS points legible instead of crushing them into the corner.
    log = axis == "diff"
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    for m in sorted(df["model"].unique()):
        for reg in ["ps", "nonps"]:
            s = df[(df.model == m) & (df.regime == reg)]
            if s.empty:
                continue
            ax.scatter(s["sim"], s["pred"], marker=markers.get(m, "o"),
                       c=colors[reg], s=55, edgecolor="k", linewidth=0.4,
                       alpha=0.85, label=f"{m} / {reg}")
    if log:
        pos = np.concatenate([df["sim"].values, df["pred"].values])
        pos = pos[pos > 0]
        span = [pos.min() * 0.7, pos.max() * 1.4]
        ax.set_xscale("log"); ax.set_yscale("log")
    else:
        hi = max(df["sim"].max(), df["pred"].max())
        lo = min(0.0, df["sim"].min(), df["pred"].min())
        span = [lo, hi + 0.08 * abs(hi - lo)]
    ax.plot(span, span, "k--", lw=1, label="y = x")
    ax.set_xlim(span); ax.set_ylim(span)
    ax.set_xlabel(f"simulated {AXES[axis]['sim_col']}")
    ax.set_ylabel(f"surrogate-predicted {AXES[axis]['sim_col']}")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def report_axis(df: pd.DataFrame, axis: str, paths_root: Path, scope: str,
                policy: str, min_corr: float) -> None:
    print(f"\n{'='*72}\n AXIS: {axis}  ({AXES[axis]['sim_col']})   n={len(df)} endpoints\n{'='*72}")
    full = breakdown(df, min_corr, per_model=True)
    with pd.option_context("display.width", 160):
        print(full.to_string(index=False))

    # worst mispredictions
    print("\n-- largest |pred - sim| --")
    show_cols = ["model", "start_idx", "regime", "pred", "sim", "endpoint_p_ps", "abs_err"]
    if "sim_psp" in df.columns:
        show_cols.insert(5, "sim_psp")
    print(df.sort_values("abs_err", ascending=False).head(5)[show_cols]
            .round(3).to_string(index=False))

    # psp confusion (eos only): model PS call vs simulated PS
    if "sim_psp" in df.columns:
        model_ps = (df["endpoint_p_ps"] >= REGIME_KEEP).astype(int)
        conf = pd.crosstab(model_ps.rename("model_calls_ps"),
                           df["sim_psp"].rename("sim_psp"))
        print("\n-- PS confusion (model endpoint_p_ps>=0.5  vs  simulated psp) --")
        print(conf.to_string())

    # ---- write outputs, co-located with beam predictions ----
    for m in sorted(df["model"].unique()):
        sub = df[df.model == m].copy()
        out_dir = paths_root / m / scope / policy
        out_dir.mkdir(parents=True, exist_ok=True)
        sub.drop(columns=["axis"]).to_csv(out_dir / f"validation_endpoints_{axis}.csv", index=False)
        breakdown(sub, min_corr, per_model=False).to_csv(
            out_dir / f"validation_summary_{axis}.csv", index=False)
        r = _corr_row("ALL", sub, min_corr)
        parity_plot(sub, axis,
                    f"{m} {scope}/{policy} — {AXES[axis]['sim_col']}  "
                    f"(r={r['pearson']}, n={r['n']})",
                    out_dir / f"validation_parity_{axis}.png")

    all_dir = paths_root / "ALL" / scope / policy
    all_dir.mkdir(parents=True, exist_ok=True)
    full.to_csv(all_dir / f"validation_summary_{axis}.csv", index=False)
    df.drop(columns=["axis"]).to_csv(all_dir / f"validation_endpoints_{axis}.csv", index=False)
    r = _corr_row("ALL", df, min_corr)
    parity_plot(df, axis,
                f"ALL {scope}/{policy} — {AXES[axis]['sim_col']}  "
                f"(r={r['pearson']}, n={r['n']})",
                all_dir / f"validation_parity_{axis}.png")
    print(f"\nwrote per-model outputs under {paths_root}/<MODEL>/{scope}/{policy}/")
    print(f"wrote aggregate outputs under {all_dir}/")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scratch_dir", required=True, type=Path,
                    help="Root of the model scratch tree holding "
                         "<MODEL>/VALIDATION/<SCOPE>/SIMULATIONS/ (cluster: $SCRATCH_AL).")
    ap.add_argument("--paths_root", type=Path, default=None,
                    help="Root of the beam PATHS tree (<MODEL>/<SCOPE>/<POLICY>/). "
                         "Defaults to <scratch_dir>/PATHS[_FIXED_LENGTH]. Locally "
                         "point at runs/beam_inspection/PATHS.")
    ap.add_argument("--models", nargs="+", default=["HPS_URRY", "MPIPI", "CALVADOS"])
    ap.add_argument("--scope", default="BENCHMARK")
    ap.add_argument("--policy", default="expert_tied")
    ap.add_argument("--length_changes", action="store_true",
                    help="Variable-length beam (PATHS). Default is PATHS_FIXED_LENGTH. "
                         "Only affects the --paths_root default.")
    ap.add_argument("--axes", nargs="+", default=["eos", "diff"], choices=list(AXES))
    ap.add_argument("--min_corr", type=float, default=0.8,
                    help="Pearson r threshold for the pass flag (default 0.8).")
    args = ap.parse_args()

    if args.paths_root is None:
        tree = "PATHS" if args.length_changes else "PATHS_FIXED_LENGTH"
        args.paths_root = args.scratch_dir / tree

    print(f"scratch_dir: {args.scratch_dir}")
    print(f"paths_root:  {args.paths_root}")
    print(f"scope/policy: {args.scope}/{args.policy}   models: {', '.join(args.models)}")

    for axis in args.axes:
        frames = [load_axis(args.scratch_dir, m, args.scope, axis) for m in args.models]
        frames = [f for f in frames if not f.empty]
        if not frames:
            print(f"\n[axis {axis}] no sim data yet "
                  f"(looked for {AXES[axis]['sim_file']} under "
                  f"<MODEL>/VALIDATION/{args.scope}/SIMULATIONS/{AXES[axis]['phase']}/) — skipping")
            continue
        report_axis(pd.concat(frames, ignore_index=True), axis,
                    args.paths_root, args.scope, args.policy, args.min_corr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
