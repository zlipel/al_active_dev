"""Diagnose beam-search benchmark results after a `run_beams.sh` cycle.

Standard workflow after each benchmark iteration:

    # 1. Sync results from cluster to local
    rsync -av stellar:/scratch/gpfs/zl4808/PROJECTS/MODEL_COMPARISON/PATHS_FIXED_LENGTH/ \\
        runs/beam_inspection/PATHS_FIXED_LENGTH/

    # 2. Run this script against the synced tree
    python beam_search/tools/analyze_benchmark.py \\
        --paths_root runs/beam_inspection/PATHS_FIXED_LENGTH

Reports (per model):
  1. Viability summary — hit / regime-kept / viable / close-miss / far-miss
  2. Termination breakdown — how many finished_quantile vs no_finished, average
     step counts, average final best-distance, per-reason improvement trend.
  3. no_finished per-target detail — final_bdsf, stagnation counter, whether
     patience or max_steps caused the stop.

Reads two artifact classes per (model, start) pair:
  - RESULTS/start_XXXX/paths.csv           end-of-walk hit/miss + regime info
  - step_timings/start_XXXX.csv            per-step convergence trace (only
                                            populated when the walk ran with
                                            ``--profile``)

Column meanings (paths.csv):
  hit                 True if the walk landed a candidate in the ±tol box
  reason              finished_quantile / no_finished / no_valid_candidates
  du_ach, dv_ach      achieved delta from start to endpoint in (u, v)
  endpoint_p_ps       final regime score — used to test regime preservation
  start_regime        'ps' or 'nonps' — starting regime

Column meanings (step_timings/start_XXXX.csv, from --profile):
  best_dist_so_far    L2 quantile distance of the beam's best candidate so far
  stagnant_steps      consecutive non-improving steps at time of that row
  n_finished, n_beam  beam bookkeeping — number of finished/active nodes

The three-way viable / close / far bucket is intentionally lenient — targets
with |miss| <= 0.015 are treated as "sim-worthy near-miss" candidates that
could still validate the surrogate's directional prediction even if they
didn't hit the strict tolerance box.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


REGIME_KEEP_THRESHOLD = 0.5   # p_ps split for regime preservation
NEAR_MISS_MAX = 0.015         # miss <= this is a sim-worthy near-miss
STAGNANT_PATIENCE = 20        # matches init_beams.sh default
MAX_STEPS_HINT = 99           # if n_steps >= this, likely capped by max_steps


def _load_paths(paths_root: Path, models: list[str], policy: str) -> pd.DataFrame:
    frames = []
    for m in models:
        base = paths_root / m / "BENCHMARK" / policy / "RESULTS"
        if not base.exists():
            print(f"[warn] {base} missing — skipping {m}", file=sys.stderr)
            continue
        for p in base.glob("start_*/paths.csv"):
            df = pd.read_csv(p)
            df["model"] = m
            frames.append(df)
    if not frames:
        raise SystemExit(f"No paths.csv found under {paths_root}")
    out = pd.concat(frames, ignore_index=True)
    out["miss"] = np.hypot(out["du_ach"] - out["du_req"], out["dv_ach"] - out["dv_req"])
    return out


def _load_step_timings(paths_root: Path, models: list[str], policy: str) -> pd.DataFrame:
    rows = []
    for m in models:
        base = paths_root / m / "BENCHMARK" / policy / "step_timings"
        if not base.exists():
            continue
        for p in base.glob("start_*.csv"):
            df = pd.read_csv(p)
            df = df[df["step"] > 0]   # step 0 = start-warmup predict, no progress
            if "best_dist_so_far" not in df.columns:
                # timings from a pre-progress_out build — nothing to analyze
                continue
            for (sidx, dur, dvr), sub in df.groupby(["start_idx","du_req","dv_req"], sort=False):
                sub = sub.sort_values("step").reset_index(drop=True)
                n_steps = len(sub)
                rows.append(dict(
                    model=m,
                    start_idx=int(sidx),
                    du_req=float(dur),
                    dv_req=float(dvr),
                    n_steps=n_steps,
                    final_bdsf=float(sub["best_dist_so_far"].iloc[-1]),
                    min_bdsf=float(sub["best_dist_so_far"].min()),
                    final_stagn=int(sub["stagnant_steps"].iloc[-1]),
                    final_nfin=int(sub["n_finished"].iloc[-1]),
                    improvement_last20=(
                        float(sub["best_dist_so_far"].iloc[-20]
                              - sub["best_dist_so_far"].iloc[-1])
                        if n_steps >= 20 else float("nan")
                    ),
                ))
    return pd.DataFrame(rows)


def viability_summary(paths: pd.DataFrame) -> None:
    """Per-model buckets: hit / regime-kept / viable / near-miss / far-miss."""
    print(f"\n{'model':<10} {'total':>6} {'hit':>5} {'reg_kept':>9} "
          f"{'viable':>7} {'near_miss':>10} {'far_miss':>9}")
    tot = defaultdict(int)
    for m in paths["model"].unique():
        sub = paths[paths["model"] == m]
        total = len(sub)
        hit = int((sub["hit"] == True).sum())     # noqa: E712 (CSV bool as string)
        kept = 0; viable = 0; near = 0; far = 0
        for _, r in sub.iterrows():
            hit_b = r["hit"] == True
            kept_b = (
                (r["endpoint_p_ps"] >= REGIME_KEEP_THRESHOLD)
                if r["start_regime"] == "ps"
                else (r["endpoint_p_ps"] < REGIME_KEEP_THRESHOLD)
            )
            kept += int(kept_b)
            viable += int(hit_b and kept_b)
            if not hit_b:
                if r["miss"] <= NEAR_MISS_MAX: near += 1
                else:                          far += 1
        print(f"{m:<10} {total:>6} {hit:>5} {kept:>9} "
              f"{viable:>7} {near:>10} {far:>9}")
        for k, v in [("total", total), ("hit", hit), ("kept", kept),
                     ("viable", viable), ("near", near), ("far", far)]:
            tot[k] += v
    n = tot["total"]
    print(f"\nviable (hit + regime kept):    {tot['viable']}/{n} = {tot['viable']/n:.0%}")
    print(f"near-miss (|miss| <= {NEAR_MISS_MAX}):    {tot['near']}/{n} = {tot['near']/n:.0%}")
    print(f"sim-worthy (viable + near):    {tot['viable']+tot['near']}/{n} = "
          f"{(tot['viable']+tot['near'])/n:.0%}")


def convergence_summary(paths: pd.DataFrame, timings: pd.DataFrame, tol_axis: float) -> None:
    if timings.empty:
        print("\n[no convergence traces found — re-run beam with --profile to enable]")
        return

    tol_l2 = np.sqrt(2) * tol_axis
    key = ["model","start_idx","du_req","dv_req"]
    merged = timings.merge(
        paths.rename(columns={"start_idx":"start_idx"})[key + ["hit","reason","miss","endpoint_p_ps"]],
        on=key, how="left",
    )

    print(f"\nTOL_L2 (hit radius) = {tol_l2:.5f}  (axis tol = {tol_axis})")
    print(f"Total targets: {len(merged)}, hits: {int((merged['hit']==True).sum())}")

    print("\n--- Termination breakdown ---")
    g = merged.groupby("reason").agg(
        n=("start_idx","count"),
        avg_steps=("n_steps","mean"),
        avg_final_bdsf=("final_bdsf","mean"),
        avg_stagn=("final_stagn","mean"),
        avg_imp_last20=("improvement_last20","mean"),
    ).round(5)
    print(g)

    nf = merged[merged["reason"] == "no_finished"].copy()
    if nf.empty:
        return

    nf["stopped_by"] = np.where(
        nf["final_stagn"] >= STAGNANT_PATIENCE, "patience",
        np.where(nf["n_steps"] >= MAX_STEPS_HINT, "max_steps", "other")
    )
    print(f"\n--- no_finished stopped_by ({len(nf)} targets) ---")
    print(nf["stopped_by"].value_counts().to_string())
    print(f"\navg final_bdsf: {nf['final_bdsf'].mean():.5f}  ({nf['final_bdsf'].mean()/tol_l2:.1f}× tol)")
    print(f"min final_bdsf: {nf['final_bdsf'].min():.5f}  ({nf['final_bdsf'].min()/tol_l2:.1f}× tol)")
    print(f"max final_bdsf: {nf['final_bdsf'].max():.5f}  ({nf['final_bdsf'].max()/tol_l2:.1f}× tol)")

    print("\n--- Per-target detail (no_finished only, sorted by model then final_bdsf) ---")
    cols = ["model","start_idx","du_req","dv_req","n_steps","final_bdsf",
            "final_stagn","improvement_last20","stopped_by"]
    show = nf[cols].sort_values(["model","final_bdsf"]).round(5)
    print(show.to_string(index=False))

    print("\n--- Interpretation aid ---")
    stuck_early = int(((nf["stopped_by"]=="patience") & (nf["improvement_last20"] == 0)).sum())
    stuck_late  = int(((nf["stopped_by"]=="patience") & (nf["improvement_last20"] > 0)).sum())
    print(f"  Genuinely stuck (patience + 0 last-20 improvement): {stuck_early}")
    print(f"    → Increasing patience won't help. Bottleneck is landscape/surrogate.")
    print(f"  Late improvement (patience + nonzero last-20 improvement): {stuck_late}")
    print(f"    → Could benefit from higher patience.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--paths_root", required=True, type=Path,
        help="Local root of the synced benchmark tree "
             "(e.g. runs/beam_inspection/PATHS_FIXED_LENGTH).",
    )
    ap.add_argument(
        "--models", nargs="+", default=["HPS_URRY","MPIPI","CALVADOS"],
        help="Models to include in the report.",
    )
    ap.add_argument(
        "--policy", default="expert_tied",
        help="Policy subfolder under BENCHMARK/ (default: expert_tied).",
    )
    ap.add_argument(
        "--tol_axis", type=float, default=0.00625,
        help="Per-axis quantile tolerance used by the beam. Default matches "
             "init_beams.sh (grid_spacing/2 = 0.00625). Only affects the "
             "convergence report's TOL_L2 label.",
    )
    args = ap.parse_args()

    paths = _load_paths(args.paths_root, args.models, args.policy)
    timings = _load_step_timings(args.paths_root, args.models, args.policy)

    print(f"=== Benchmark viability report ===")
    print(f"Root: {args.paths_root}")
    print(f"Policy: {args.policy}")
    print(f"Models: {', '.join(args.models)}")
    viability_summary(paths)
    convergence_summary(paths, timings, args.tol_axis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
