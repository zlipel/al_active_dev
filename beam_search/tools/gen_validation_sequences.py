"""Generate seq_<scope>.txt + endpoints_metadata.csv from beam benchmark results.

Runs on the cluster login node. For each model, reads paths.csv from the beam
benchmark RESULTS/ tree, filters to viable + near-miss endpoints, dedups on
end_seq, and writes to the validation SIMULATIONS/ tree that make_eos.sh /
make_diff.sh --validation <SCOPE> pick up.

Output layout (per model):

    $SCRATCH_AL/<MODEL>/VALIDATION/<SCOPE>/SIMULATIONS/
        ├── EOS/
        │   ├── seq_<scope-lower>.txt        one unique end_seq per line
        │   └── endpoints_metadata.csv       row-per-endpoint join key
        └── DIFF/
            ├── seq_<scope-lower>.txt        (same file, mirrored)
            └── endpoints_metadata.csv

The metadata CSV carries a ``sim_row`` column pointing at the seq_<scope>.txt
line index — after LAMMPS finishes, join sim outputs to the beam-side
(start_idx, du_req, dv_req) tuple via sim_row.

Usage:
    python beam_search/tools/gen_validation_sequences.py \\
        --scratch_dir $SCRATCH_AL \\
        --length_changes \\
        --scope BENCHMARK

Passing --include_far_miss also emits far-miss endpoints for surrogate stress
testing (useful when you want to see how badly predictions degrade past the
near-miss threshold).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REGIME_KEEP = 0.5
NEAR_MISS_MAX = 0.015


def _load_endpoints(paths_root: Path, model: str, policy: str) -> pd.DataFrame:
    base = paths_root / model / "BENCHMARK" / policy / "RESULTS"
    if not base.exists():
        return pd.DataFrame()
    frames = []
    for p in sorted(base.glob("start_*/paths.csv")):
        df = pd.read_csv(p)
        df["model"] = model
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["miss"] = np.hypot(out["du_ach"] - out["du_req"], out["dv_ach"] - out["dv_req"])
    return out


def _classify(row) -> str:
    hit = bool(row["hit"])
    if row["start_regime"] == "ps":
        kept = row["endpoint_p_ps"] >= REGIME_KEEP
    else:
        kept = row["endpoint_p_ps"] < REGIME_KEEP
    if hit and kept:
        return "viable"
    if hit and not kept:
        return "hit_regime_slip"
    if not hit and row["miss"] <= NEAR_MISS_MAX:
        return "near_miss"
    return "far_miss"


def _write_for_model(
    scratch_dir: Path,
    model: str,
    scope: str,
    keep_categories: set[str],
    endpoints: pd.DataFrame,
) -> int:
    endpoints = endpoints.copy()
    endpoints["category"] = endpoints.apply(_classify, axis=1)
    keep = endpoints[endpoints["category"].isin(keep_categories)].reset_index(drop=True)
    if keep.empty:
        print(f"[{model}] no endpoints match categories {sorted(keep_categories)} — skipping")
        return 0

    # Drop rows with missing end_seq (no_valid_candidates cases).
    n_before = len(keep)
    keep = keep.dropna(subset=["end_seq"]).reset_index(drop=True)
    if len(keep) != n_before:
        print(f"[{model}] dropped {n_before - len(keep)} endpoints with no end_seq")

    # sim_row = index into the deduplicated unique-sequence list.
    unique_seqs: list[str] = list(dict.fromkeys(keep["end_seq"].astype(str)))
    keep["sim_row"] = keep["end_seq"].astype(str).map({s: i for i, s in enumerate(unique_seqs)})

    meta_cols = [
        "sim_row", "category", "model", "start_idx", "du_req", "dv_req",
        "start_seq", "end_seq", "start_regime", "start_p_ps",
        "rho_target", "diff_target", "rho_end", "diff_end",
        "u_end", "v_end", "du_ach", "dv_ach", "miss", "endpoint_p_ps",
        "hit", "reason",
    ]
    # Preserve only the columns actually present (paths.csv schema is stable
    # but future column additions should not break this script).
    meta_cols = [c for c in meta_cols if c in keep.columns]
    meta = keep[meta_cols].sort_values(["sim_row", "start_idx", "du_req", "dv_req"]).reset_index(drop=True)

    scope_lower = scope.lower()
    n_written = 0
    for phase in ("EOS", "DIFF"):
        out_dir = scratch_dir / model / "VALIDATION" / scope / "SIMULATIONS" / phase
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"seq_{scope_lower}.txt").write_text("\n".join(unique_seqs) + "\n")
        meta.to_csv(out_dir / "endpoints_metadata.csv", index=False)
        n_written += 1

    dist = keep["category"].value_counts().to_dict()
    print(f"[{model}] {len(unique_seqs)} unique sequences ({len(keep)} endpoints, {dist}) "
          f"→ VALIDATION/{scope}/SIMULATIONS/{{EOS,DIFF}}/")
    return len(unique_seqs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--scratch_dir", required=True, type=Path,
        help="Root of the model scratch tree (e.g. $SCRATCH_AL = "
             "/scratch/gpfs/USER/PROJECTS/MODEL_COMPARISON).",
    )
    ap.add_argument(
        "--models", nargs="+", default=["HPS_URRY", "MPIPI", "CALVADOS"],
        help="Models to build seq files for.",
    )
    ap.add_argument("--policy", default="expert_tied")
    ap.add_argument(
        "--scope", default="BENCHMARK",
        help="Subfolder under VALIDATION/. Convention: uppercase. "
             "The written seq file becomes seq_<scope-lower>.txt.",
    )
    ap.add_argument(
        "--length_changes", action="store_true",
        help="Read from PATHS/ (default reads PATHS_FIXED_LENGTH/).",
    )
    ap.add_argument(
        "--include_far_miss", action="store_true",
        help="Also include far-miss endpoints (|miss| > 0.015). Default: "
             "viable + near-miss only (the sim-worthy set).",
    )
    args = ap.parse_args()

    paths_root = args.scratch_dir / ("PATHS" if args.length_changes else "PATHS_FIXED_LENGTH")
    if not paths_root.exists():
        raise SystemExit(f"Beam-results root not found: {paths_root}")

    keep_categories = {"viable", "near_miss"}
    if args.include_far_miss:
        keep_categories.add("far_miss")

    print(f"Reading from: {paths_root}")
    print(f"Writing to:   {args.scratch_dir}/<MODEL>/VALIDATION/{args.scope}/SIMULATIONS/{{EOS,DIFF}}/")
    print(f"Keeping:      {sorted(keep_categories)}")

    total = 0
    for model in args.models:
        endpoints = _load_endpoints(paths_root, model, args.policy)
        if endpoints.empty:
            print(f"[{model}] no paths.csv found — skipping")
            continue
        total += _write_for_model(
            args.scratch_dir, model, args.scope, keep_categories, endpoints,
        )

    print(f"\nDone. Total unique sequences across models: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
