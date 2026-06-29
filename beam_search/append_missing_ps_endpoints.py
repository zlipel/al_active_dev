#!/usr/bin/env python3
"""
append_missing_phase_separating_endpoints.py

Finds phase-separating start sequences absent from (or partially covered in)
the existing endpoints CSV, and appends the missing endpoint rows.

Phase-separating: density > --density_min (default 0.0).

Detection is at the endpoint-key level (start_idx, du_req, dv_req), so starts
with partial grids are handled correctly.

Example:
    python append_missing_phase_separating_endpoints.py \\
        --model CALVADOS \\
        --scratch_dir /scratch/gpfs/zl4808/PROJECTS/MODEL_COMPARISON \\
        --home_dir    /home/zl4808/PROJECTS/MODEL_COMPARISON \\
        --db_root     /home/zl4808/scripts/GENDATA \\
        --final_iter  8 \\
        --dry_run
"""
import os
import sys
import argparse
import shutil
import glob
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from shapely.geometry import MultiPoint, Point

# ---------- delta grid (must match prepare_endpoints.py exactly) ----------
_DELTA_VALS = np.array([-0.05, -0.04, -0.03, -0.02, 0.0, 0.02, 0.03, 0.04, 0.05])
DIRECTIONS = [
    (du, dv)
    for du in _DELTA_VALS
    for dv in _DELTA_VALS
    if not (abs(du) < 1e-12 and abs(dv) < 1e-12)
]
assert len(DIRECTIONS) == 80, f"Expected 80 directions, got {len(DIRECTIONS)}"

DIRECTION_SET = {(round(float(du), 8), round(float(dv), 8)) for du, dv in DIRECTIONS}

DEDUP_KEY = ["start_idx", "du_req", "dv_req"]

ENDPOINT_COLS = [
    "model", "start_idx", "start_seq",
    "u_start", "v_start", "rho_start", "diff_start",
    "rho_start_yjz", "diff_start_yjz",
    "du_req", "dv_req", "u_target", "v_target", "inside_hull",
]


def _rk(du, dv):
    return (round(float(du), 8), round(float(dv), 8))


def paths_dir_for(scratch_dir, model, length_changes):
    base = "PATHS" if length_changes else "PATHS_FIXED_LENGTH"
    return os.path.join(scratch_dir, base, model)


def labels_csv_path(scratch_dir, model, final_iter):
    return os.path.join(
        scratch_dir, model, "GENERATIONS",
        f"iteration_{final_iter}", f"labels_gen{final_iter}.csv",
    )


def validate_labels_csv(lcsv, n_seqs):
    """Guardrail 2: labels CSV must have required columns, correct row count, no NaN."""
    if not os.path.exists(lcsv):
        raise FileNotFoundError(f"Labels CSV not found: {lcsv}")
    df = pd.read_csv(lcsv)
    required = {"density", "exp_density", "diff"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Labels CSV missing columns: {missing_cols}")
    if len(df) != n_seqs:
        raise ValueError(
            f"Labels CSV has {len(df)} rows but bundle has {n_seqs} sequences — must match."
        )
    for col in required:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            raise ValueError(f"Labels CSV column '{col}' has {n_nan} NaN values.")
    return df


def load_bundle(home_dir, scratch_dir, db_root, model, final_iter,
                front="upper", ehvi_variant="epsilon",
                exploration_strategy="kriging_believer", transform="yeoj", mc_ehvi=False):
    from cross_paths.model_io import load_all_models
    from al_pipeline.core.paths import ALPaths
    paths = ALPaths(
        base_path=home_dir,
        scratch_path=scratch_dir,
        iteration=final_iter,
        front=front,
        model=model,
        ehvi_variant=ehvi_variant,
        exploration_strategy=exploration_strategy,
        transform=transform,
        mc_ehvi=mc_ehvi,
    )
    return load_all_models(paths, db_dir=os.path.join(db_root, "databases"))[model]


def build_quantile_transforms(bundle):
    rho  = bundle.labels[:, 0]
    diff = bundle.labels[:, 1]
    q_rho = QuantileTransformer(
        n_quantiles=min(1000, len(rho)), random_state=0, output_distribution="uniform"
    ).fit(rho.reshape(-1, 1))
    q_diff = QuantileTransformer(
        n_quantiles=min(1000, len(diff)), random_state=0, output_distribution="uniform"
    ).fit(diff.reshape(-1, 1))
    return q_rho, q_diff


def build_hull_and_uv(bundle, q_rho, q_diff):
    rho  = bundle.labels[:, 0]
    diff = bundle.labels[:, 1]
    u = q_rho.transform(rho.reshape(-1, 1))[:, 0]
    v = q_diff.transform(diff.reshape(-1, 1))[:, 0]
    hull = MultiPoint(np.column_stack([u, v])).convex_hull
    return hull, u, v


def get_covered_keys(existing_df):
    """For each start_idx, which (du_req, dv_req) keys exist (rounded to 8 dp)."""
    covered = defaultdict(set)
    if existing_df.empty:
        return covered
    for row in existing_df[["start_idx", "du_req", "dv_req"]].itertuples(index=False):
        covered[int(row.start_idx)].add(_rk(row.du_req, row.dv_req))
    return covered


def generate_rows(ps_indices, covered, bundle, q_rho, q_diff, hull, u_all, v_all, model):
    """Return (new_rows, stats) for all missing endpoint keys."""
    s_rho, s_diff = bundle.label_scalers
    new_rows = []
    n_full = n_partial = n_none = 0

    for raw_idx in ps_indices:
        idx = int(raw_idx)
        existing_dirs = covered.get(idx, set())
        missing_dirs  = DIRECTION_SET - existing_dirs

        if not missing_dirs:
            n_full += 1
            continue
        if not existing_dirs:
            n_none += 1
        else:
            n_partial += 1

        u0   = float(u_all[idx])
        v0   = float(v_all[idx])
        rho0 = float(bundle.labels[idx, 0])
        diff0= float(bundle.labels[idx, 1])
        seq0 = bundle.sequences[idx]
        rho_yjz  = float(s_rho.transform([[rho0]])[0, 0])
        diff_yjz = float(s_diff.transform([[diff0]])[0, 0])

        for du_req, dv_req in DIRECTIONS:
            if _rk(du_req, dv_req) not in missing_dirs:
                continue
            u_t = float(np.clip(u0 + du_req, 0.0, 1.0))
            v_t = float(np.clip(v0 + dv_req, 0.0, 1.0))
            new_rows.append({
                "model":          model,
                "start_idx":      idx,
                "start_seq":      seq0,
                "u_start":        u0,
                "v_start":        v0,
                "rho_start":      rho0,
                "diff_start":     diff0,
                "rho_start_yjz":  rho_yjz,
                "diff_start_yjz": diff_yjz,
                "du_req":         du_req,
                "dv_req":         dv_req,
                "u_target":       u_t,
                "v_target":       v_t,
                "inside_hull":    hull.covers(Point(u_t, v_t)),
            })

    stats = {"n_already_full": n_full, "n_fully_missing": n_none, "n_partial": n_partial}
    return new_rows, stats


def backup(endpoints_path):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = endpoints_path + f".bak_{ts}"
    shutil.copy2(endpoints_path, dst)
    print(f"[backup]  {endpoints_path}")
    print(f"       -> {dst}")
    return dst


def write_output(existing_df, new_rows, endpoints_path, pdir, model, dry_run, no_backup):
    if not new_rows:
        print("\n[result] No new rows to append — endpoints file unchanged.")
        return

    new_df = pd.DataFrame(new_rows, columns=ENDPOINT_COLS)

    if dry_run:
        print(f"\n[dry_run] Would append {len(new_df)} rows to:\n  {endpoints_path}")
        n_inside = int(new_df["inside_hull"].sum())
        print(f"[dry_run] inside_hull: {n_inside}/{len(new_df)}")
        return

    # Guardrail 3: backup before writing (default on; suppress with --no_backup)
    if os.path.exists(endpoints_path) and not no_backup:
        backup(endpoints_path)

    # Merge: existing rows win on key conflict (keep="first")
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=DEDUP_KEY, keep="first")
    combined.to_csv(endpoints_path, index=False)
    print(f"\n[write]   {endpoints_path}")
    print(f"          {len(combined)} total rows (+{len(new_df)} appended)")

    # Summary CSV
    summary_path = os.path.join(pdir, f"append_summary_{model}.csv")
    new_df[["start_idx", "du_req", "dv_req", "inside_hull"]].to_csv(summary_path, index=False)
    print(f"[summary] {summary_path}")

    # Appended indices TXT
    idx_path = os.path.join(pdir, f"appended_start_indices_{model}.txt")
    appended = sorted(new_df["start_idx"].unique())
    with open(idx_path, "w") as f:
        f.write("\n".join(str(i) for i in appended) + "\n")
    print(f"[indices] {idx_path} ({len(appended)} starts appended)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model",       required=True, choices=["HPS_URRY", "MPIPI", "CALVADOS"])
    ap.add_argument("--scratch_dir", required=True)
    ap.add_argument("--home_dir",    required=True)
    ap.add_argument("--db_root",     required=True)
    ap.add_argument("--final_iter",  type=int, default=8)
    ap.add_argument("--density_min", type=float, default=0.0,
                    help="density threshold for phase-separating (default 0.0)")
    ap.add_argument("--length_changes", action="store_true",
                    help="Use PATHS/ instead of PATHS_FIXED_LENGTH/")
    ap.add_argument("--dry_run",  action="store_true",
                    help="Report counts but do not write anything")
    ap.add_argument("--no_backup", action="store_true",
                    help="Skip backup of existing endpoints CSV before writing")
    # ALPaths construction args — needed to resolve GPR checkpoint with correct naming
    ap.add_argument("--front",                default="upper")
    ap.add_argument("--ehvi_variant",         default="epsilon")
    ap.add_argument("--exploration_strategy", default="kriging_believer")
    ap.add_argument("--transform",            default="yeoj")
    ap.add_argument("--mc_ehvi", action="store_true")
    args = ap.parse_args()

    pdir           = paths_dir_for(args.scratch_dir, args.model, args.length_changes)
    endpoints_path = os.path.join(pdir, f"endpoints_{args.model}.csv")
    lcsv           = labels_csv_path(args.scratch_dir, args.model, args.final_iter)

    print(f"[config] model={args.model}  final_iter={args.final_iter}  density_min={args.density_min}")
    print(f"[config] endpoints : {endpoints_path}")
    print(f"[config] labels_csv: {lcsv}")
    print(f"[config] dry_run={args.dry_run}  no_backup={args.no_backup}")
    print()

    # Step 1: load bundle
    print("[step 1] Loading model bundle...")
    bundle = load_bundle(args.home_dir, args.scratch_dir, args.db_root, args.model, args.final_iter,
                         front=args.front, ehvi_variant=args.ehvi_variant,
                         exploration_strategy=args.exploration_strategy,
                         transform=args.transform, mc_ehvi=args.mc_ehvi)
    print(f"         {len(bundle.sequences)} sequences loaded")

    # Step 2: validate labels CSV (guardrail 2)
    print("[step 2] Validating labels CSV...")
    labels_df = validate_labels_csv(lcsv, len(bundle.sequences))
    print(f"         OK — {len(labels_df)} rows, columns: {list(labels_df.columns)}")

    # Step 3: quantile transforms + hull
    print("[step 3] Building quantile transforms and convex hull...")
    q_rho, q_diff = build_quantile_transforms(bundle)
    hull, u_all, v_all = build_hull_and_uv(bundle, q_rho, q_diff)

    # Step 4: identify phase-separating starts
    density    = labels_df["density"].values
    ps_mask    = density > args.density_min
    ps_indices = np.where(ps_mask)[0]
    print(f"[step 4] Phase-separating starts (density > {args.density_min}): {len(ps_indices)}")

    # Step 5: load existing endpoints (append-only; file must exist)
    print("[step 5] Loading existing endpoints...")
    if not os.path.exists(endpoints_path):
        raise FileNotFoundError(
            f"Existing endpoint file not found: {endpoints_path}. "
            "This script is append-only; run prepare_endpoints.py first "
            "or check --length_changes / --model / --scratch_dir."
        )

    existing_df = pd.read_csv(endpoints_path)
    print(f"         {len(existing_df)} rows found")
    # Coverage check (endpoint-key level, guardrail 1)
    covered = get_covered_keys(existing_df)
    n_fully_covered = sum(
        1 for idx in ps_indices
        if len(covered.get(int(idx), set())) == 80
    )
    print(f"         PS starts already fully covered: {n_fully_covered}/{len(ps_indices)}")

    # Resume summary: inspect RESULTS folders (guardrail 4)
    results_base = os.path.join(pdir, "RESULTS")
    if os.path.isdir(results_base):
        result_csvs = glob.glob(os.path.join(results_base, "start_*", "paths.csv"))
        ps_set = {int(i) for i in ps_indices}
        ps_with_endpoint_rows = {int(i) for i in ps_indices if int(i) in covered}
        print(f"\n[resume] RESULTS/start_*/paths.csv files present: {len(result_csvs)}")
        print(f"[resume] PS starts with any existing endpoint rows: {len(ps_with_endpoint_rows)}")

    # Step 6: generate missing rows
    print("\n[step 6] Generating missing endpoint rows...")
    new_rows, stats = generate_rows(
        ps_indices, covered, bundle, q_rho, q_diff, hull, u_all, v_all, args.model
    )
    print(f"         PS starts already complete  : {stats['n_already_full']}")
    print(f"         PS starts fully missing      : {stats['n_fully_missing']}")
    print(f"         PS starts partially missing  : {stats['n_partial']}")
    print(f"         New endpoint rows to append  : {len(new_rows)}")

    write_output(existing_df, new_rows, endpoints_path, pdir, args.model, args.dry_run, args.no_backup)


if __name__ == "__main__":
    main()
