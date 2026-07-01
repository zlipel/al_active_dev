"""
Shared helpers used by both retrospective (`al_retrospective.py`) and forward
(`al_forward.py`) diagnostics.

Every helper here operates on the completed-run artifacts (cumulative
features_gen{N}.csv / labels_gen{N}.csv / seq_gen{N}.txt) or wraps the
per-iter surrogate refit pattern with tempdir-scoped configs. Nothing here
knows about acquisition (EHVI) or scoring.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from pygmo import hypervolume

from al_pipeline.core.config import ALConfig
from al_pipeline.data_prep.parents import find_pareto_front
from al_pipeline.ga.ga_utils import load_models, load_moe_bundle, load_normalization_stats
from al_pipeline.surrogates import GlobalGPRSurrogate, MoESurrogate, Surrogate
from al_pipeline.training.kfold_training import train_from_config


# ---------- data model ----------

@dataclass(frozen=True)
class IterationData:
    """
    All rows across all iters of a completed AL run.

    features_df / labels_df / seqs are cumulative and aligned by row index —
    row `i` in features_df is the raw features for `seqs[i]` with labels at
    `labels_df.iloc[i]`. The `generation` column in labels_df identifies
    which iter each row was simulated at.
    """
    features_df: pd.DataFrame
    labels_df: pd.DataFrame
    seqs: list[str]

    def __post_init__(self) -> None:
        n = len(self.labels_df)
        if len(self.features_df) != n or len(self.seqs) != n:
            raise ValueError(
                f"features/labels/seqs mis-aligned: "
                f"len(features)={len(self.features_df)}, len(labels)={n}, len(seqs)={len(self.seqs)}"
            )
        if "generation" not in self.labels_df.columns:
            raise ValueError("labels_df must have a 'generation' column")

    def training_slice_before(self, gen: int) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        """Rows where `generation < gen` — what the AL loop had at iter `gen`."""
        mask = self.labels_df["generation"] < gen
        return (
            self.features_df.iloc[mask.values].reset_index(drop=True),
            self.labels_df.iloc[mask.values].reset_index(drop=True),
            [s for s, m in zip(self.seqs, mask.values) if m],
        )

    def proposal_pool_at(self, gen: int) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        """Rows where `generation == gen` — the real batch of children at iter `gen`."""
        mask = self.labels_df["generation"] == gen
        return (
            self.features_df.iloc[mask.values].reset_index(drop=True),
            self.labels_df.iloc[mask.values].reset_index(drop=True),
            [s for s, m in zip(self.seqs, mask.values) if m],
        )


def load_completed_run(runs_root: Path, model: str, n_iters: int) -> IterationData:
    """
    Read the final iter's cumulative artifacts. features/labels/seqs are all
    written cumulatively by the AL loop, so we only need the last iter's files.
    """
    runs_root = Path(runs_root)
    gen_dir = runs_root / model / "GENERATIONS" / f"iteration_{n_iters}"
    features_path = gen_dir / f"features_gen{n_iters}.csv"
    labels_path = gen_dir / f"labels_gen{n_iters}.csv"
    seq_path = gen_dir / f"seq_gen{n_iters}.txt"
    for pth in (features_path, labels_path, seq_path):
        if not pth.exists():
            raise FileNotFoundError(f"Missing completed-run artifact: {pth}")

    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)
    with open(seq_path) as f:
        seqs = [ln.strip() for ln in f if ln.strip()]
    return IterationData(features_df=features_df, labels_df=labels_df, seqs=seqs)


# ---------- target front ----------

def compute_target_front(labels_df: pd.DataFrame, front: str, obj1: str, obj2: str) -> np.ndarray:
    """
    Pareto front from the union of all completed iters — the retrospective's
    "target". Returned as an (N, 2) array in raw objective space with columns
    (obj1, obj2), one row per Pareto member.
    """
    kind = ["max", "max"] if front == "upper" else ["min", "min"]
    front_df, _idx = find_pareto_front(
        labels_df[[obj1, obj2]].reset_index(drop=True),
        kind=kind,
        objectives=[obj1, obj2],
    )
    return front_df[[obj1, obj2]].to_numpy()


# ---------- HV in raw objective space ----------

def _raw_pareto_ref_point(pmax: np.ndarray, pmin: np.ndarray, margin: float = 0.05) -> np.ndarray:
    """Ref point strictly worse than every observed point (in MIN space)."""
    return pmax + margin * np.abs(pmax - pmin) + 1e-9


def compute_hv_raw(labels_df: pd.DataFrame, front: str, obj1: str, obj2: str,
                    ref_point_min: np.ndarray) -> float:
    """
    HV of `labels_df`'s Pareto front in RAW objective space.

    Both objectives get flipped to MIN space (pygmo convention). For an
    'upper' front (max-max in raw), we negate both columns; for 'lower',
    we leave them.
    """
    pts = labels_df[[obj1, obj2]].to_numpy(dtype=np.float64)
    if front == "upper":
        pts = -pts
    _, idx = find_pareto_front(
        pd.DataFrame(pts, columns=[obj1, obj2]),
        kind=["min", "min"], objectives=[obj1, obj2],
    )
    frontier = pts[idx]
    frontier = frontier[np.all(frontier < ref_point_min, axis=1)]
    if len(frontier) == 0:
        return 0.0
    return float(hypervolume(frontier).compute(ref_point_min))


def _global_ref_point(all_labels_df: pd.DataFrame, front: str, obj1: str, obj2: str,
                       margin: float = 0.05) -> np.ndarray:
    """
    A single ref point used across all iters + surrogates so HVs are directly
    comparable. Computed from the WHOLE completed run (worst point + margin).
    """
    pts = all_labels_df[[obj1, obj2]].to_numpy(dtype=np.float64)
    if front == "upper":
        pts = -pts
    return _raw_pareto_ref_point(pts.max(axis=0), pts.min(axis=0), margin=margin)


# ---------- per-iter surrogate fitting via tempdirs ----------

def _make_iter_cfg(
    tempdir: Path,
    cfg_base: ALConfig,
    iteration: int,
    train_model_type: str,
) -> ALConfig:
    """
    Rebuild cfg with tempdir as base/scratch. Everything else preserved.
    """
    base = tempdir / "home"
    scratch = tempdir / "scratch"
    for d in (base, scratch):
        d.mkdir(parents=True, exist_ok=True)
    return replace(
        cfg_base,
        base_path=base, scratch_path=scratch,
        iteration=iteration,
        train_model_type=train_model_type,
    )


def _write_training_slice(cfg: ALConfig, feats: pd.DataFrame, labels: pd.DataFrame) -> None:
    """Materialize features_csv + labels_csv in the tempdir path scheme."""
    p = cfg.paths
    p.iter_scratch_dir.mkdir(parents=True, exist_ok=True)
    p.models_dir.mkdir(parents=True, exist_ok=True)
    feats.to_csv(p.features_csv, index=False)
    labels.to_csv(p.labels_csv, index=False)


def _build_moe_surrogates(cfg_moe: ALConfig, policies: tuple[str, ...] = ("soft", "hard")) -> dict[str, Surrogate]:
    """Train + load MoE bundle; wrap in one Surrogate per policy."""
    train_from_config(cfg_moe)
    bundle = load_moe_bundle(cfg_moe, temp=False)
    return {f"moe_{pol}": MoESurrogate(bundle, policy=pol) for pol in policies}


def _build_global_surrogate(cfg_global: ALConfig) -> Surrogate:
    """Train + load global multitask GPR; wrap in GlobalGPRSurrogate."""
    train_from_config(cfg_global)
    model_bundle = load_models(cfg_global, temp=False, device="cpu")
    normalization_stats = load_normalization_stats(cfg_global.paths.norm_stats)
    return GlobalGPRSurrogate(
        mode="gpr_multitask",
        model_bundle=model_bundle,
        normalization_stats=normalization_stats,
        obj1=cfg_global.obj1, obj2=cfg_global.obj2,
    )
