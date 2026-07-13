"""Endpoint-target preparation for beam search — benchmark + production modes.

Two selection modes off the same script, sharing:

- Regime split by ``density > 0`` (matches MoE training's PS definition —
  ``moe_training.py`` uses ``ps_col = cfg.aux1_obj1 = 'density'``).
- p_ps thresholds on the final full-data classifier to filter each regime pool
  (``PS pool = regime=ps ∩ p_ps ≥ thresh_higher``; ``nonPS pool = regime=nonps
  ∩ p_ps ≤ thresh_lower``).
- Quantile transform on ``exp_density`` (III.8: continuous across both regimes,
  unlike ``density`` which is 0 for ~76% of sequences and would degenerate the
  QT).
- Grid stratification of each pool by ``(u, v)`` property coordinates.
- Convex hull on labeled ``(u, v)`` — targets outside the hull are marked
  ``inside_hull=False``.

Mode differences:

- **benchmark** — picks ``N`` sequences per regime; one from each of ``N``
  distinct non-empty bins chosen to maximize bin-index spread (farthest-point
  in bin space). Targets: 4 diagonals at ``±benchmark_delta``.
- **production** — picks ``⌈frac × |bin|⌉`` sequences per non-empty bin.
  Targets: full symmetric grid over ``{±grid_spacing, ±2·grid_spacing, ...}``
  up to ``±largest_delta`` on each axis, excluding zero on both axes.

Outputs land under ``<scratch>/PATHS[_FIXED_LENGTH]/<MODEL>/<MODE>/`` — the
runner consumes ``endpoints_<MODEL>.csv`` from the same folder and writes
results into ``<policy>/RESULTS/``.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint, Point
from sklearn.preprocessing import QuantileTransformer

from cross_paths.model_io import load_all_models
from al_pipeline.core.paths import ALPaths


# ---------------------------------------------------------------------------
# Grid stratification helpers
# ---------------------------------------------------------------------------


def stratify_pool_by_grid(
    uv: np.ndarray,
    pool_idx: np.ndarray,
    n_bins: int,
) -> dict[tuple[int, int], np.ndarray]:
    """Bin the sequences at ``pool_idx`` by their ``(u, v)`` into ``n_bins × n_bins``.

    Returns a dict keyed by ``(i, j)`` bin indices, values are the labeled-set
    indices of sequences that fall in that bin. Empty bins are omitted so
    callers can iterate over ``dict.items()`` and skip the empties naturally.

    ``u_edges = v_edges = np.linspace(0, 1, n_bins + 1)``. The upper edge on
    the last bin is inclusive so a sequence at exactly ``u=1.0`` still lands
    somewhere — this matches ``QuantileTransformer(output_distribution=
    "uniform")``'s clip behavior at the extremes.
    """
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive; got {n_bins}")
    if len(pool_idx) == 0:
        return {}

    u_edges = np.linspace(0.0, 1.0, n_bins + 1)
    v_edges = np.linspace(0.0, 1.0, n_bins + 1)

    u_pool = uv[pool_idx, 0]
    v_pool = uv[pool_idx, 1]

    # `np.digitize` returns 1-based bin indices; subtract 1 and clip so u=1.0
    # lands in the last bin rather than an out-of-range slot.
    u_bin = np.clip(np.digitize(u_pool, u_edges) - 1, 0, n_bins - 1)
    v_bin = np.clip(np.digitize(v_pool, v_edges) - 1, 0, n_bins - 1)

    bins: dict[tuple[int, int], list[int]] = {}
    for k, idx in enumerate(pool_idx):
        key = (int(u_bin[k]), int(v_bin[k]))
        bins.setdefault(key, []).append(int(idx))

    return {k: np.asarray(v, dtype=int) for k, v in bins.items()}


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def farthest_point_bin_sample(
    non_empty_bins: list[tuple[int, int]],
    N: int,
) -> list[tuple[int, int]]:
    """Pick ``N`` bins from ``non_empty_bins`` maximizing pairwise Chebyshev spread.

    Greedy farthest-point sampling. Deterministic seed: the first pick is the
    bin whose ``(i + j)`` is smallest (corner-closest). If ``N`` exceeds the
    number of non-empty bins, cycles through the list — callers get multiple
    picks per bin only when the pool is bin-starved.
    """
    if N <= 0 or not non_empty_bins:
        return []

    remaining = list(non_empty_bins)
    # First pick: closest to bin-index origin. Deterministic and independent
    # of any RNG state.
    remaining.sort(key=lambda b: (b[0] + b[1], b[0], b[1]))
    picked = [remaining[0]]
    remaining = remaining[1:]

    while len(picked) < N and remaining:
        # Pick the remaining bin whose min-Chebyshev-distance to already-picked
        # is maximum. Ties broken by (row, col) to keep the result stable.
        best_bin = max(
            remaining,
            key=lambda b: (min(_chebyshev(b, p) for p in picked), -b[0], -b[1]),
        )
        picked.append(best_bin)
        remaining.remove(best_bin)

    if len(picked) < N:
        # Ran out of distinct bins; cycle through picked and repeat until we
        # hit N. The caller will pick a fresh sequence within the bin each time.
        cycle_idx = 0
        while len(picked) < N:
            picked.append(picked[cycle_idx % len(picked)])
            cycle_idx += 1

    return picked[:N]


def select_benchmark_starts(
    bins: dict[tuple[int, int], np.ndarray],
    N: int,
    rng: np.random.Generator,
) -> list[tuple[tuple[int, int], int]]:
    """Return ``[((u_bin, v_bin), seq_idx), ...]`` of length ``N``.

    Bin selection via ``farthest_point_bin_sample``; within-bin selection is
    uniformly random over the bin's sequences (seeded ``rng``). If the same
    bin gets picked twice (only possible when N > n_non_empty_bins), a
    different sequence is chosen without replacement — falls back to
    replacement only when the bin's own sequence count is exhausted.
    """
    if N <= 0 or not bins:
        return []

    non_empty = list(bins.keys())
    chosen_bins = farthest_point_bin_sample(non_empty, N)

    used_per_bin: dict[tuple[int, int], set[int]] = {b: set() for b in non_empty}
    out: list[tuple[tuple[int, int], int]] = []

    for bkey in chosen_bins:
        candidates = bins[bkey]
        unused = [int(i) for i in candidates if int(i) not in used_per_bin[bkey]]
        if not unused:
            # Bin exhausted — allow replacement. Only happens when N is
            # forcing multiple picks in a small bin.
            seq_idx = int(rng.choice(candidates))
        else:
            seq_idx = int(rng.choice(unused))
            used_per_bin[bkey].add(seq_idx)
        out.append((bkey, seq_idx))

    return out


def select_production_starts(
    bins: dict[tuple[int, int], np.ndarray],
    frac: float,
    rng: np.random.Generator,
) -> list[tuple[tuple[int, int], int]]:
    """Return ``[((u_bin, v_bin), seq_idx), ...]``.

    From each non-empty bin, take ``⌈frac × |bin|⌉`` sequences uniformly at
    random within the bin (no replacement). ``frac=1.0`` picks every sequence
    in every bin; ``frac=0.0`` picks none.
    """
    if frac <= 0.0 or not bins:
        return []

    out: list[tuple[tuple[int, int], int]] = []
    for bkey, members in bins.items():
        n_pick = int(math.ceil(frac * len(members)))
        n_pick = min(n_pick, len(members))
        if n_pick == 0:
            continue
        chosen = rng.choice(members, size=n_pick, replace=False)
        for seq_idx in chosen:
            out.append((bkey, int(seq_idx)))

    return out


# ---------------------------------------------------------------------------
# Target-delta grid
# ---------------------------------------------------------------------------


def make_target_deltas(
    mode: str,
    *,
    grid_spacing: float,
    largest_delta: float,
    benchmark_delta: float,
) -> list[tuple[float, float]]:
    """Return the list of ``(du, dv)`` offsets in quantile space.

    Benchmark: 4 diagonals at ``(±benchmark_delta, ±benchmark_delta)``.

    Production: symmetric grid ``{-largest_delta, ..., -grid_spacing,
    +grid_spacing, ..., +largest_delta}`` per axis, excluding zero on both
    axes. All cells are true 2D moves. Grid size = ``(2·k)² = 4·k²`` where
    ``k = round(largest_delta / grid_spacing)``. Defaults give ``(2·4)² =
    64`` cells.
    """
    if mode == "benchmark":
        d = float(benchmark_delta)
        return [(+d, +d), (+d, -d), (-d, +d), (-d, -d)]

    if mode == "production":
        if grid_spacing <= 0 or largest_delta <= 0:
            raise ValueError(
                f"grid_spacing and largest_delta must be positive; got "
                f"{grid_spacing=}, {largest_delta=}"
            )
        k = int(round(largest_delta / grid_spacing))
        if k < 1:
            raise ValueError(
                f"largest_delta ({largest_delta}) < grid_spacing "
                f"({grid_spacing}) — no cells to search"
            )
        pos = [round(grid_spacing * (i + 1), 6) for i in range(k)]
        vals = [-v for v in pos[::-1]] + pos
        return [(du, dv) for du in vals for dv in vals]

    raise ValueError(f"unknown mode={mode!r}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()

    # ---- required infra ----
    parser.add_argument("--scratch_dir", required=True)
    parser.add_argument("--home_dir",    required=True)
    parser.add_argument("--db_root",     required=True)
    parser.add_argument("--model",       required=True,
                        choices=["HPS_URRY", "MPIPI", "CALVADOS"])
    parser.add_argument("--final_iter",  type=int, default=10)
    parser.add_argument("--length_changes", action="store_true",
                        help="Allow length-changing edits in neighbor enumeration")

    # ---- mode ----
    parser.add_argument("--mode", choices=["benchmark", "production"],
                        default="benchmark",
                        help="Selection + target-grid regime. Benchmark: small "
                             "N per regime, 4-diagonal targets. Production: "
                             "frac-of-pool per bin, full symmetric grid.")

    # ---- p_ps thresholds (shared) ----
    parser.add_argument("--thresh_lower",  type=float, default=0.25,
                        help="nonPS pool = regime=nonps ∩ p_ps ≤ thresh_lower")
    parser.add_argument("--thresh_higher", type=float, default=0.75,
                        help="PS pool = regime=ps ∩ p_ps ≥ thresh_higher")

    # ---- grid stratification (shared) ----
    parser.add_argument("--ps_bins",     type=int, default=3,
                        help="Grid size on (u, v) for PS pool stratification")
    parser.add_argument("--nonps_bins",  type=int, default=5,
                        help="Grid size on (u, v) for nonPS pool stratification")

    # ---- target grid (shared parameters, mode picks which ones apply) ----
    parser.add_argument("--grid_spacing",     type=float, default=0.0125,
                        help="Production target grid step in quantile space")
    parser.add_argument("--largest_delta",    type=float, default=0.05,
                        help="Production target grid max |Δ| per axis")
    parser.add_argument("--benchmark_delta",  type=float, default=0.0375,
                        help="Benchmark diagonal-target step in quantile space")

    # ---- benchmark-only ----
    parser.add_argument("--n_ps",    type=int, default=2,
                        help="Benchmark: number of PS starts")
    parser.add_argument("--n_nonps", type=int, default=2,
                        help="Benchmark: number of nonPS starts")

    # ---- production-only ----
    parser.add_argument("--frac_ps",    type=float, default=0.9,
                        help="Production: fraction of each PS bin sampled")
    parser.add_argument("--frac_nonps", type=float, default=0.75,
                        help="Production: fraction of each nonPS bin sampled")

    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for within-bin picks + QT construction")

    # ---- ALPaths ----
    parser.add_argument("--front",                default="upper")
    parser.add_argument("--ehvi_variant",         default="epsilon")
    parser.add_argument("--exploration_strategy", default="kriging_believer")
    parser.add_argument("--transform",            default="yeoj")
    parser.add_argument("--mc_ehvi", action="store_true")

    args = parser.parse_args()

    if args.length_changes:
        print("Preparing endpoints allowing length-changing edits!!", flush=True)

    rng = np.random.default_rng(args.seed)

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

    rho = bundle.labels_exp_density
    diff = bundle.labels_diff
    density = bundle.labels_density
    start_regime = bundle.start_regime
    seqs = bundle.sequences
    rho_yjz_scaler, diff_yjz_scaler = bundle.label_scalers

    # p_ps under the final full-data gate for every labeled sequence.
    p_ps_all = bundle.surrogate.predict_design(bundle.features).p_ps
    if p_ps_all is None:
        # Global surrogate has no gate; fill NaN so downstream columns still write.
        p_ps_all = np.full(len(rho), np.nan, dtype=np.float64)

    # --- quantile transform on exp_density + diff (III.8) ---
    q_rho = QuantileTransformer(
        n_quantiles=min(1000, len(rho)),
        random_state=args.seed,
        output_distribution="uniform",
    ).fit(rho.reshape(-1, 1))
    q_diff = QuantileTransformer(
        n_quantiles=min(1000, len(diff)),
        random_state=args.seed,
        output_distribution="uniform",
    ).fit(diff.reshape(-1, 1))

    u = q_rho.transform(rho.reshape(-1, 1))[:, 0]
    v = q_diff.transform(diff.reshape(-1, 1))[:, 0]
    uv = np.column_stack([u, v])
    hull = MultiPoint(uv).convex_hull

    # --- regime + p_ps thresholding to build the two pools ---
    ps_mask = start_regime & (p_ps_all >= args.thresh_higher)
    nonps_mask = (~start_regime) & (p_ps_all <= args.thresh_lower)
    ps_pool_idx = np.flatnonzero(ps_mask)
    nonps_pool_idx = np.flatnonzero(nonps_mask)

    # --- bin each pool by (u, v) ---
    ps_bins = stratify_pool_by_grid(uv, ps_pool_idx, args.ps_bins)
    nonps_bins = stratify_pool_by_grid(uv, nonps_pool_idx, args.nonps_bins)

    # --- select starts per mode ---
    if args.mode == "benchmark":
        ps_picks = select_benchmark_starts(ps_bins, args.n_ps, rng)
        nonps_picks = select_benchmark_starts(nonps_bins, args.n_nonps, rng)
    else:  # production
        ps_picks = select_production_starts(ps_bins, args.frac_ps, rng)
        nonps_picks = select_production_starts(nonps_bins, args.frac_nonps, rng)

    all_picks: list[tuple[str, tuple[int, int], int]] = (
        [("ps", b, i) for b, i in ps_picks]
        + [("nonps", b, i) for b, i in nonps_picks]
    )

    # De-dup by seq_idx (safety net against replacement fallback in benchmark mode).
    seen: set[int] = set()
    unique_picks: list[tuple[str, tuple[int, int], int]] = []
    for regime, bkey, seq_idx in all_picks:
        if seq_idx in seen:
            continue
        seen.add(seq_idx)
        unique_picks.append((regime, bkey, seq_idx))

    # --- starts CSV rows ---
    starts_rows = []
    for regime, bkey, seq_idx in unique_picks:
        starts_rows.append({
            "model":   args.model,
            "idx":     int(seq_idx),
            "seq":     seqs[int(seq_idx)],
            "regime":  regime,
            "rho":     float(rho[int(seq_idx)]),
            "diff":    float(diff[int(seq_idx)]),
            "density": float(density[int(seq_idx)]),
            "p_ps":    float(p_ps_all[int(seq_idx)]),
            "u":       float(u[int(seq_idx)]),
            "v":       float(v[int(seq_idx)]),
            "u_bin":   bkey[0],
            "v_bin":   bkey[1],
        })
    starts_df = pd.DataFrame(starts_rows)

    # --- target-delta grid ---
    directions = make_target_deltas(
        args.mode,
        grid_spacing=args.grid_spacing,
        largest_delta=args.largest_delta,
        benchmark_delta=args.benchmark_delta,
    )

    # --- endpoints CSV rows ---
    rows = []
    for regime, bkey, seq_idx in unique_picks:
        idx = int(seq_idx)
        u0, v0 = float(uv[idx, 0]), float(uv[idx, 1])
        rho0, diff0 = float(rho[idx]), float(diff[idx])
        density0 = float(density[idx])
        seq0 = seqs[idx]
        p_ps0 = float(p_ps_all[idx])

        for du_req, dv_req in directions:
            u_t = float(np.clip(u0 + du_req, 0.0, 1.0))
            v_t = float(np.clip(v0 + dv_req, 0.0, 1.0))
            rows.append({
                "model":          args.model,
                "start_idx":      idx,
                "start_seq":      seq0,
                "start_regime":   regime,
                "start_p_ps":     p_ps0,
                "start_density":  density0,
                "u_start":        u0,
                "v_start":        v0,
                "rho_start":      rho0,
                "diff_start":     diff0,
                "rho_start_yjz":  float(rho_yjz_scaler.transform([[rho0]])[0, 0]),
                "diff_start_yjz": float(diff_yjz_scaler.transform([[diff0]])[0, 0]),
                "du_req":         float(du_req),
                "dv_req":         float(dv_req),
                "u_target":       u_t,
                "v_target":       v_t,
                "inside_hull":    bool(hull.covers(Point(u_t, v_t))),
            })
    endpoints_df = pd.DataFrame(rows)

    # --- output layout: <scratch>/PATHS[_FIXED_LENGTH]/<MODEL>/<MODE>/ ---
    out_dir = os.path.join(
        args.scratch_dir,
        "PATHS" if args.length_changes else "PATHS_FIXED_LENGTH",
        args.model,
        args.mode.upper(),
    )
    os.makedirs(out_dir, exist_ok=True)

    endpoints_path = os.path.join(out_dir, f"endpoints_{args.model}.csv")
    starts_path    = os.path.join(out_dir, f"starts_{args.model}.csv")
    config_path    = os.path.join(out_dir, "config.json")

    endpoints_df.to_csv(endpoints_path, index=False)
    starts_df.to_csv(starts_path, index=False)

    # --- config.json — thresholds, pool sizes, bin usage, per-start summaries ---
    config = {
        "mode":       args.mode,
        "model":      args.model,
        "final_iter": args.final_iter,
        "seed":       args.seed,
        "thresholds": {
            "lower":  args.thresh_lower,
            "higher": args.thresh_higher,
        },
        "stratification": {
            "ps_bins":         args.ps_bins,
            "nonps_bins":      args.nonps_bins,
            "ps_pool_size":    int(ps_pool_idx.size),
            "nonps_pool_size": int(nonps_pool_idx.size),
            "ps_bins_used":    len(ps_bins),
            "nonps_bins_used": len(nonps_bins),
            "n_ps_selected":    sum(1 for r, *_ in unique_picks if r == "ps"),
            "n_nonps_selected": sum(1 for r, *_ in unique_picks if r == "nonps"),
        },
        "targets": {
            "grid_spacing":       args.grid_spacing,
            "largest_delta":      args.largest_delta,
            "benchmark_delta":    args.benchmark_delta,
            "n_targets_per_start": len(directions),
        },
        "mode_params": (
            {"n_ps": args.n_ps, "n_nonps": args.n_nonps}
            if args.mode == "benchmark"
            else {"frac_ps": args.frac_ps, "frac_nonps": args.frac_nonps}
        ),
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    n_ps_sel = config["stratification"]["n_ps_selected"]
    n_nonps_sel = config["stratification"]["n_nonps_selected"]
    print(f"[{args.mode}] Wrote endpoints → {endpoints_path}")
    print(f"[{args.mode}] Wrote starts    → {starts_path}")
    print(f"[{args.mode}] Wrote config    → {config_path}")
    print(f"[{args.mode}] Selected {n_ps_sel} PS + {n_nonps_sel} nonPS starts "
          f"({len(directions)} targets each)")


if __name__ == "__main__":
    main()
