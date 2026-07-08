"""Endpoint-target preparation for beam search.

Row 8 rewrite (feat/beam-surrogate-cleanup):

- Loads the beam bundle through the new `load_beam_bundle` API (surrogate +
  labels + density + featurizer), removing the custom loader that refit label
  scalers at load time.
- Loads `density` alongside `exp_density` and `diff`; writes `start_regime`
  (``density > 0``), ``start_density``, and ``start_p_ps`` (under the final
  full-data gate) into the endpoints CSV so downstream scoring can filter
  by regime without re-deriving.
- Keeps the production 5×5 quantile stratification path unchanged from the
  pre-refactor version. The diagnostic-scope p_ps-based start selection
  (§III.2) will be wired in the next branch (feat/beam-policy) alongside the
  policy argument.

The quantile transformer is fit on **`exp_density`** (III.8: continuous
across both regimes) — not on `density`, which is 0 for the majority of
labeled sequences and would degenerate the QT. The regime split is on
`density`; the two axes serve different purposes and are both recorded.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint, Point
from sklearn.preprocessing import QuantileTransformer

from cross_paths.model_io import load_all_models
from al_pipeline.core.paths import ALPaths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch_dir", required=True)
    parser.add_argument("--home_dir",    required=True)
    parser.add_argument("--db_root",     required=True)
    parser.add_argument("--model",       required=True, choices=["HPS_URRY", "MPIPI", "CALVADOS"])
    parser.add_argument("--final_iter",  type=int, default=8)
    parser.add_argument("--n_bins",      type=int, default=10)  # stratification bins per axis
    parser.add_argument("--k_per_bin",   type=int, default=1)
    parser.add_argument("--length_changes", action="store_true",
                        help="Allow length-changing edits in neighbor enumeration")
    # ALPaths construction args — needed to resolve MoE checkpoints with correct naming
    parser.add_argument("--front",                default="upper")
    parser.add_argument("--ehvi_variant",         default="epsilon")
    parser.add_argument("--exploration_strategy", default="kriging_believer")
    parser.add_argument("--transform",            default="yeoj")
    parser.add_argument("--mc_ehvi", action="store_true")
    args = parser.parse_args()

    if args.length_changes:
        print("Preparing endpoints allowing length-changing edits!!", flush=True)

    al_paths = ALPaths(
        base_path=args.home_dir,
        scratch_path=args.scratch_dir,
        iteration=args.final_iter,
        front=args.front,
        model=args.model,
        ehvi_variant=args.ehvi_variant,
        exploration_strategy=args.exploration_strategy,
        transform=args.transform,
        mc_ehvi=args.mc_ehvi,
    )

    bundles = load_all_models(al_paths, db_dir=os.path.join(args.db_root, "databases"))
    bundle = bundles[args.model]

    rho = bundle.labels_exp_density                     # (N,) physical
    diff = bundle.labels_diff                           # (N,) physical
    density = bundle.labels_density                     # (N,) physical (>0 iff PS)
    start_regime = bundle.start_regime                  # (N,) bool
    seqs = bundle.sequences
    rho_yjz_scaler, diff_yjz_scaler = bundle.label_scalers

    # p_ps under the final full-data gate for every labeled sequence. Used
    # both as a per-start column in the endpoints CSV and (in a later
    # branch) as the diagnostic-scope start selector.
    p_ps_all = bundle.surrogate.predict_design(bundle.features).p_ps
    if p_ps_all is None:
        # Global surrogate wouldn't populate p_ps; MoE always does. Guard so a
        # future --policy=global path degrades gracefully.
        p_ps_all = np.full(len(rho), np.nan, dtype=np.float64)

    # --- quantile transform on exp_density (III.8) ---
    q_rho = QuantileTransformer(
        n_quantiles=min(1000, len(rho)),
        random_state=0,
        output_distribution="uniform",
    ).fit(rho.reshape(-1, 1))

    q_diff = QuantileTransformer(
        n_quantiles=min(1000, len(diff)),
        random_state=0,
        output_distribution="uniform",
    ).fit(diff.reshape(-1, 1))

    u = q_rho.transform(rho.reshape(-1, 1))[:, 0]
    v = q_diff.transform(diff.reshape(-1, 1))[:, 0]
    uv = np.column_stack([u, v])
    hull = MultiPoint(uv).convex_hull

    # --- stratified start selection in quantile space (production path) ---
    rng = np.random.default_rng(0)
    u_all = uv[:, 0]
    v_all = uv[:, 1]
    chosen_idx: list[int] = []

    u_edges = np.linspace(0.0, 1.0, args.n_bins + 1)
    v_edges = np.linspace(0.0, 1.0, args.n_bins + 1)

    stratified_meta = []
    for i in range(args.n_bins):
        for j in range(args.n_bins):
            u_min, u_max = u_edges[i], u_edges[i + 1]
            v_min, v_max = v_edges[j], v_edges[j + 1]
            # include right edge on last bin to avoid dropping points at 1.0
            u_mask = (u_all >= u_min) & (u_all <= u_max) if i == args.n_bins - 1 else (u_all >= u_min) & (u_all < u_max)
            v_mask = (v_all >= v_min) & (v_all <= v_max) if j == args.n_bins - 1 else (v_all >= v_min) & (v_all < v_max)
            bin_indices = np.where(u_mask & v_mask)[0]
            if len(bin_indices) == 0:
                continue
            k = min(args.k_per_bin, len(bin_indices))
            i_picks = rng.choice(bin_indices, size=k, replace=False)
            for i_pick in i_picks:
                chosen_idx.append(int(i_pick))
                stratified_meta.append({
                    "model":    args.model,
                    "idx":      int(i_pick),
                    "seq":      seqs[int(i_pick)],
                    "rho":      rho[int(i_pick)],
                    "diff":     diff[int(i_pick)],
                    "density":  density[int(i_pick)],
                    "regime":   "ps" if bool(start_regime[int(i_pick)]) else "nonps",
                    "p_ps":     float(p_ps_all[int(i_pick)]),
                    "u":        u_all[int(i_pick)],
                    "v":        v_all[int(i_pick)],
                    "u_bin":    i,
                    "v_bin":    j,
                })

    start_indices = np.array(sorted(set(chosen_idx)))

    # --- delta-grid in quantile space ---
    delta_vals = np.array([-0.05, -0.04, -0.03, -0.02, 0.0, 0.02, 0.03, 0.04, 0.05])
    directions = [
        (du, dv)
        for du in delta_vals
        for dv in delta_vals
        if not (abs(du) < 1e-12 and abs(dv) < 1e-12)
    ]

    rows = []
    for idx in start_indices:
        idx = int(idx)
        u0, v0 = uv[idx]
        rho0, diff0 = rho[idx], diff[idx]
        density0 = density[idx]
        seq0 = seqs[idx]
        regime0 = "ps" if bool(start_regime[idx]) else "nonps"
        p_ps0 = float(p_ps_all[idx])

        for du_req, dv_req in directions:
            u_t = float(np.clip(u0 + du_req, 0.0, 1.0))
            v_t = float(np.clip(v0 + dv_req, 0.0, 1.0))
            rows.append({
                "model":          args.model,
                "start_idx":      idx,
                "start_seq":      seq0,
                "start_regime":   regime0,
                "start_p_ps":     p_ps0,
                "start_density":  float(density0),
                "u_start":        u0,
                "v_start":        v0,
                "rho_start":      rho0,
                "diff_start":     diff0,
                "rho_start_yjz":  float(rho_yjz_scaler.transform([[rho0]])[0, 0]),
                "diff_start_yjz": float(diff_yjz_scaler.transform([[diff0]])[0, 0]),
                "du_req":         du_req,
                "dv_req":         dv_req,
                "u_target":       u_t,
                "v_target":       v_t,
                "inside_hull":    hull.covers(Point(u_t, v_t)),
            })

    endpoints_df = pd.DataFrame(rows)

    out_dir = (
        os.path.join(args.scratch_dir, "PATHS", args.model)
        if args.length_changes
        else os.path.join(args.scratch_dir, "PATHS_FIXED_LENGTH", args.model)
    )
    os.makedirs(out_dir, exist_ok=True)

    endpoints_path = os.path.join(out_dir, f"endpoints_{args.model}.csv")
    endpoints_df.to_csv(endpoints_path, index=False)

    stratified_path = os.path.join(out_dir, f"starts_stratified_{args.model}.csv")
    pd.DataFrame(stratified_meta).to_csv(stratified_path, index=False)

    n_ps = int(np.sum(start_regime[start_indices]))
    n_nonps = len(start_indices) - n_ps
    print(f"Wrote endpoints to {endpoints_path}")
    print(f"Wrote stratified starts to {stratified_path}")
    print(f"Total distinct starts: {len(start_indices)} ({n_ps} PS, {n_nonps} nonPS)")


if __name__ == "__main__":
    main()
