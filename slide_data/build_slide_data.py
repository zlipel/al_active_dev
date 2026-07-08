"""Curate MoE-track diagnostic data into tidy CSVs for the slide-figure pass.

Run under the `torch-protein-M1` conda env from repo root:

    python slide_data/build_slide_data.py

Outputs (all under `slide_data/`):
  - master_metrics.csv    : long/tidy metrics across all four diagnostics
  - regime_scatter.csv    : one row per labeled sequence at iter 10 (for slide 1)
  - callouts.csv          : specific failure-mode NLL numbers for the callout slide
  - MANIFEST.md           : sources, commit, column dict, blank-cell reasons

Every value is emitted with FULL precision (no rounding).
Genuinely-N/A cells are the empty string, never 0.
"""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "slide_data"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["CALVADOS", "HPS_URRY", "MPIPI"]
RETRO_STARTS = [1, 3, 5, 7]

# Canonical predictor tokens (per user spec). Source-file tokens map to these.
PREDICTOR_MAP_REGIME_OOF = {
    "global": "global",
    "moe_soft": "moe_soft",
    "ps_expert": "ps_expert",
    "nonps_expert": "nonps_expert",
    # moe_hard and ps_guarded are evaluated at 4 thresholds each; the
    # threshold is folded into the token below (e.g. moe_hard_t050,
    # ps_guarded_050) so predictor is unique within (model, split, property).
}


def _regime_oof_predictor(predictor: str, threshold) -> str:
    """Map a (predictor, threshold) row to the canonical token."""
    if predictor in PREDICTOR_MAP_REGIME_OOF:
        return PREDICTOR_MAP_REGIME_OOF[predictor]
    if predictor == "moe_hard":
        return f"moe_hard_t{int(round(float(threshold) * 100)):03d}"
    if predictor == "ps_guarded":
        return f"ps_guarded_{int(round(float(threshold) * 100)):03d}"
    return predictor
PREDICTOR_MAP_FORWARD = {
    "global": "global",
    "moe_soft": "moe_soft",
    "moe_hard_t015": "moe_hard_t015",
    "moe_hard_t030": "moe_hard_t030",
    "moe_hard_t050": "moe_hard_t050",
    "moe_hard_t070": "moe_hard_t070",
}

# split token: source → canonical
SPLIT_MAP = {"all": "all", "ps": "ps", "nonps": "nonps", "PS": "ps", "nonPS": "nonps"}


def _rows_from_retrospective() -> list[dict]:
    """Emit rows for hv_actual/hv_moe_soft/hv_moe_hard/hv_global per iter,
    plus one row per predictor for target_hv (constant per file) and
    rounds_to_95pct (dict of int|None)."""
    rows: list[dict] = []
    for model in MODELS:
        for start in RETRO_STARTS:
            f = REPO / "runs" / model / "DIAGNOSTIC" / f"retrospective_trajectory_upper_start{start}.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            iters = d["iters"]
            front = d.get("front", "upper")
            for key, pred in [
                ("hv_actual", "actual"),
                ("hv_moe_soft", "moe_soft"),
                ("hv_moe_hard", "moe_hard"),
                ("hv_global", "global"),
            ]:
                for it, val in zip(iters, d[key]):
                    rows.append(dict(
                        diagnostic="retrospective", model=model,
                        start_iter=start, iteration=it,
                        predictor=pred, property="", split="", front=front,
                        metric="hv", value=val,
                    ))
            for pred in ("actual", "moe_soft", "moe_hard", "global"):
                rows.append(dict(
                    diagnostic="retrospective", model=model,
                    start_iter=start, iteration="",
                    predictor=pred, property="", split="", front=front,
                    metric="target_hv", value=d["target_hv"],
                ))
            r2t = d["rounds_to_95pct"]
            key_by_pred = {"actual": "actual", "moe_soft": "moe_soft",
                           "moe_hard": "moe_hard", "global": "global"}
            for pred, key in key_by_pred.items():
                val = r2t.get(key)
                rows.append(dict(
                    diagnostic="retrospective", model=model,
                    start_iter=start, iteration="",
                    predictor=pred, property="", split="", front=front,
                    metric="rounds_to_95pct",
                    value="" if val is None else int(val),
                ))
    return rows


def _rows_from_forward() -> list[dict]:
    """Long forward metrics: (heldout_iter, predictor, property, space, split,
    front_type) → r2/rmse/spearman/nll_z. We keep space='z' rows only (that is
    the space the slide narrative uses; physical-space rows are redundant for
    R²/Spearman and duplicate for NLL after scaling). Front stratification
    kept so callers can pick upper vs. lower."""
    rows: list[dict] = []
    for model in MODELS:
        f = REPO / "runs" / model / "DIAGNOSTIC" / "forward_metrics_start1.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df = df[df.space == "z"]
        for _, r in df.iterrows():
            pred = PREDICTOR_MAP_FORWARD.get(r.predictor, r.predictor)
            split = SPLIT_MAP.get(r.split, r.split)
            for metric in ("r2", "rmse", "spearman", "nll_z"):
                val = r[metric]
                if pd.isna(val):
                    val = ""
                rows.append(dict(
                    diagnostic="forward", model=model,
                    start_iter=1, iteration=int(r.heldout_iter),
                    predictor=pred, property=r.property, split=split,
                    front="" if r.front_type == "all" else r.front_type,
                    metric=metric, value=val,
                ))
    return rows


def _rows_from_forward_classifier() -> list[dict]:
    """RF gate metrics per held-out iter, per front_type."""
    rows: list[dict] = []
    for model in MODELS:
        f = REPO / "runs" / model / "DIAGNOSTIC" / "forward_classifier_start1.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        for _, r in df.iterrows():
            for src, out_metric in [
                ("roc_auc", "roc_auc"),
                ("f1", "f1"),
                ("ps_recall", "ps_recall"),
                ("nonps_fpr", "nonps_fpr"),
            ]:
                val = r[src]
                if pd.isna(val):
                    val = ""
                rows.append(dict(
                    diagnostic="classifier", model=model,
                    start_iter=1, iteration=int(r.heldout_iter),
                    predictor="", property="", split="",
                    front="" if r.front_type == "all" else r.front_type,
                    metric=out_metric, value=val,
                ))
    return rows


def _rows_from_regime_oof() -> list[dict]:
    """Terminal-iter OOF at iter 10. Keeps ONLY split ∈ {all, PS, nonPS} to
    honor the tidy schema; the finer PS_*_q25/q75 and p_ps_* sub-splits are
    dropped (documented in MANIFEST). Space='z' only."""
    rows: list[dict] = []
    for model in MODELS:
        f = REPO / "runs" / model / "DIAGNOSTIC" / "regime_oof_metrics_iter10.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df = df[df.space == "z"]
        df = df[df.split.isin(["all", "PS", "nonPS"])]
        for _, r in df.iterrows():
            pred = _regime_oof_predictor(r.predictor, r.threshold)
            split = SPLIT_MAP.get(r.split, r.split)
            for metric in ("r2", "rmse", "spearman", "nll_z"):
                val = r[metric]
                if pd.isna(val):
                    val = ""
                rows.append(dict(
                    diagnostic="regime_oof", model=model,
                    start_iter="", iteration=10,
                    predictor=pred, property=r.property, split=split,
                    front="",
                    metric=metric, value=val,
                ))
        # classifier metrics (single-row CSV)
        cf = REPO / "runs" / model / "DIAGNOSTIC" / "regime_oof_classifier_iter10.csv"
        if cf.exists():
            cdf = pd.read_csv(cf)
            for _, cr in cdf.iterrows():
                for src, out_metric in [
                    ("roc_auc", "roc_auc"),
                    ("f1", "f1"),
                    ("recall", "ps_recall"),
                    ("brier", "brier"),
                ]:
                    val = cr[src]
                    if pd.isna(val):
                        val = ""
                    rows.append(dict(
                        diagnostic="classifier", model=model,
                        start_iter="", iteration=10,
                        predictor="", property="", split="",
                        front="",
                        metric=out_metric, value=val,
                    ))
    return rows


COLUMNS = ["diagnostic", "model", "start_iter", "iteration",
           "predictor", "property", "split", "front", "metric", "value"]


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})


def build_master_metrics() -> tuple[int, dict]:
    rows: list[dict] = []
    rows.extend(_rows_from_retrospective())
    rows.extend(_rows_from_forward())
    rows.extend(_rows_from_forward_classifier())
    rows.extend(_rows_from_regime_oof())
    _write_csv(OUT / "master_metrics.csv", rows, COLUMNS)
    tally: dict = {}
    for r in rows:
        tally[r["diagnostic"]] = tally.get(r["diagnostic"], 0) + 1
    return len(rows), tally


def build_regime_scatter() -> int:
    """Row per sequence at iter 10 (600 rows per model, spanning gens 0..10).

    Sequences have no persistent seq_id in the source files; we use the
    zero-based row index within the labels CSV as `seq_id`. features_gen10 has
    the same row order so `seq_id` also indexes the feature matrix if a caller
    ever needs it. `density` is the LAMMPS coexistence density (0 for nonPS);
    `regime` is defined as (density > 0) — the same rule MoE training uses
    (moe_training.py:335)."""
    header = ["model", "seq_id", "generation", "exp_density", "diffusivity",
              "density", "regime"]
    out_rows: list[dict] = []
    for model in MODELS:
        f = (Path("/Users/zl4808/Documents/ActiveLearningAndIDPs/PROJECTS/MODEL_COMPARISON")
             / model / "GENERATIONS" / "iteration_10" / "labels_gen10.csv")
        df = pd.read_csv(f)
        for i, r in df.iterrows():
            out_rows.append(dict(
                model=model, seq_id=int(i),
                generation=int(r.generation),
                exp_density=r.exp_density,
                diffusivity=r["diff"],
                density=r.density,
                regime="PS" if float(r.density) > 0 else "nonPS",
            ))
    _write_csv(OUT / "regime_scatter.csv", out_rows, header)
    return len(out_rows)


def build_callouts() -> int:
    """Two conditions:

    1. ps_only_expert_mixed_test — regime_oof at iter 10, predictor=ps_expert,
       split=all, space=z. One row per (model, property). regime=blank
       (mixed test). tau=blank. Consumers wanting the "single number per model"
       cited in MOE_SUMMARY.md take the mean across properties (verified:
       (214.85 + 11.10)/2 = 112.98 for CALVADOS).

    2. hard_tau — forward_ranking_start1.csv aggregate. One row per
       (model, τ, regime) using mean_NLL_z_ps and mean_NLL_z_nonps. Property
       column blank because forward_ranking aggregates across both.

    Schema extension: user's spec had no `property` column; we add it so the
    ps_only rows are unambiguous. Documented in MANIFEST."""
    header = ["model", "condition", "regime", "tau", "property", "nll_z"]
    out_rows: list[dict] = []
    # ps_only expert on mixed test
    for model in MODELS:
        f = REPO / "runs" / model / "DIAGNOSTIC" / "regime_oof_metrics_iter10.csv"
        df = pd.read_csv(f)
        sub = df[(df.predictor == "ps_expert") & (df.split == "all")
                 & (df.space == "z")]
        for _, r in sub.iterrows():
            out_rows.append(dict(
                model=model,
                condition="ps_only_expert_mixed_test",
                regime="", tau="", property=r.property, nll_z=r.nll_z,
            ))
    # hard-tau NLL per regime
    for model in MODELS:
        f = REPO / "runs" / model / "DIAGNOSTIC" / "forward_ranking_start1.csv"
        df = pd.read_csv(f)
        for _, r in df.iterrows():
            if not str(r.predictor).startswith("moe_hard_"):
                continue
            tau = float(r.hard_threshold)
            out_rows.append(dict(
                model=model, condition="hard_tau",
                regime="PS", tau=tau, property="",
                nll_z=r.mean_NLL_z_ps,
            ))
            out_rows.append(dict(
                model=model, condition="hard_tau",
                regime="nonPS", tau=tau, property="",
                nll_z=r.mean_NLL_z_nonps,
            ))
    _write_csv(OUT / "callouts.csv", out_rows, header)
    return len(out_rows)


def _git_head() -> tuple[str, str]:
    sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                  text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"],
                                    text=True).strip()
    return sha, ("dirty" if dirty else "clean")


def _source_mtimes() -> dict[str, str]:
    """Modification timestamp of one canonical file per diagnostic per model,
    so MANIFEST can report the run date of the underlying data."""
    files: dict[str, Path] = {}
    for m in MODELS:
        base = REPO / "runs" / m / "DIAGNOSTIC"
        files[f"{m}/forward_metrics"] = base / "forward_metrics_start1.csv"
        files[f"{m}/forward_classifier"] = base / "forward_classifier_start1.csv"
        files[f"{m}/regime_oof_metrics"] = base / "regime_oof_metrics_iter10.csv"
        files[f"{m}/retrospective_start1"] = base / "retrospective_trajectory_upper_start1.json"
    out = {}
    for k, p in files.items():
        if p.exists():
            ts = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
            out[k] = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            out[k] = "MISSING"
    return out


def main() -> None:
    n_master, tally = build_master_metrics()
    n_scatter = build_regime_scatter()
    n_callouts = build_callouts()
    sha, state = _git_head()
    mtimes = _source_mtimes()
    print(f"master_metrics.csv rows: {n_master}  breakdown: {tally}")
    print(f"regime_scatter.csv rows: {n_scatter}")
    print(f"callouts.csv rows: {n_callouts}")
    print(f"git HEAD: {sha}  ({state})")
    print(f"source mtimes: {json.dumps(mtimes, indent=2)}")


if __name__ == "__main__":
    main()
