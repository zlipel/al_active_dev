"""
AL retrospective — counterfactual re-selection analysis for MoE vs global GPR.

Goal (per project_moe_diagnostics_plan.md): given a completed 10-round global-AL
run, ask *at each iter N, would MoE have picked better children than global did*?
Reported as (a) cumulative HV trajectory and (b) rounds-to-target HV.

Framing: **kriging-believer re-selection**. The proposal pool at each iter is
the real batch of children that ran through LAMMPS in the completed run, with
their real measured labels. At each iter we refit MoE + global on the exact
data slice the AL loop had, then run a KB inner loop that mirrors production
`cli/child.py::run_child`:

  * pick highest-EHVI child from the remaining pool;
  * predict its z-labels via the surrogate's mean (apply pessimism penalty
    from iter >= pessimism_start_iter, seq_id > 1 — same rule production uses);
  * re-condition the surrogate on those predicted labels (global: retrain the
    multitask GP with warm start + 1 Adam step; MoE: hard-gate + reindex the
    assigned expert). The RF gate stays frozen inside a batch, same as
    production augmentation;
  * update the running Pareto front with the picked child's predicted labels;
  * repeat for `k_pick` seq_ids.

At the end of the iter, HV is computed in raw physical space using the REAL
labels of the picked children — never the KB-imputed values.

Reuses (from ga/augmentation.py):
  - `predict_for_augmentation(surrogate, feats_raw_df, return_std)`
  - `overlap_batch(mu, S, mu_cands, S_cands)` — pessimism penalty
  - `_reindex_expert(assigned, feats_raw_df, z_labels, lr)` — MoE expert
    re-conditioning (identical to production `augment()`)
  - `_retrain_model_gpr_multitask(cfg, model, likelihood, X, y)` — global GP
    re-conditioning
And from ga/ga_utils.py:
  - `make_epsilon_shifted_front(cfg, front, feats_raw_df, surrogate)` —
    ε·σ̄·epsilon_scale shift applied whenever cfg.ehvi_variant == 'epsilon'.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from pygmo import hypervolume

from al_pipeline.acquisition import ehvi
from al_pipeline.core.config import ALConfig
from al_pipeline.data_prep.data_loading import convert_and_normalize_features
from al_pipeline.data_prep.parents import find_pareto_front
from al_pipeline.ga.augmentation import (
    _reindex_expert,
    _retrain_model_gpr_multitask,
    overlap_batch,
    predict_for_augmentation,
)
from al_pipeline.ga.ga_utils import (
    load_models, load_moe_bundle, load_normalization_stats,
    make_epsilon_shifted_front,
)
from al_pipeline.surrogates import (
    GlobalGPRSurrogate, MoESurrogate, Surrogate,
    build_rf_features, classifier_p_ps,
)
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
    # find_pareto_front returns the front in [obj1_key_from_kwargs, obj2_...] column order.
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
    # Filter dominated so pygmo doesn't do redundant work.
    _, idx = find_pareto_front(
        pd.DataFrame(pts, columns=[obj1, obj2]),
        kind=["min", "min"], objectives=[obj1, obj2],
    )
    frontier = pts[idx]
    # pygmo also requires every point strictly better than the ref.
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


# ---------- EHVI scoring ----------

def score_children_ehvi(
    surrogate: Surrogate,
    children_features_raw_df: pd.DataFrame,
    pareto_front_norm: np.ndarray,
    front: str,
    ref_mode: str = "frac",
    ref_frac: float = 0.5,
) -> np.ndarray:
    """
    Score each child's EHVI under `surrogate` against `pareto_front_norm`
    (in the surrogate's own normalized objective space).

    NOTE: `pareto_front_norm` is expected to be the FINAL front used for EHVI
    — if `cfg.ehvi_variant == 'epsilon'`, the caller must have already applied
    `make_epsilon_shifted_front` to shift the front outward. This function
    itself does not know about `ehvi_variant`; it's just the acquisition math.

    Returns a (B,) array of EHVI values. Higher is better.
    """
    pool = surrogate.predict_pool(children_features_raw_df)
    means, stds = pool.means, pool.stds
    augmented_front = ehvi.front_augmentation(
        pareto_front_norm, front=front,
        ref_mode=ref_mode, frac=ref_frac,
    )
    pred1, pred2 = means[:, 0], means[:, 1]
    std1, std2 = stds[:, 0], stds[:, 1]
    if front == "upper":
        # ehvi_analytic operates in MIN space; negate for upper front.
        return ehvi.ehvi_analytic(-pred1, std1, -pred2, std2, augmented_front)
    return ehvi.ehvi_analytic(pred1, std1, pred2, std2, augmented_front)


# ---------- kriging-believer re-conditioning ----------

def _recondition_surrogate(
    surrogate: Surrogate,
    cfg: ALConfig,
    new_feats_raw_df: pd.DataFrame,
    new_z_labels: np.ndarray,
) -> None:
    """
    Kriging-believer step: append the picked child's raw features + predicted
    z-space labels to the surrogate's training state, mirroring what
    production's `augmentation.augment()` does at each seq_id.

    - `MoESurrogate`: RF-hard-gate the child, then `_reindex_expert` on the
      assigned expert (RF frozen inside the batch, matching production).
    - `GlobalGPRSurrogate` (multitask only): warm-start `_retrain_model_gpr_multitask`
      on the expanded (X, y) tensors and swap the retrained model + likelihood
      into the surrogate's bundle. Single-task path isn't wired here (it was
      never used with the retrospective) — raise a clear error.

    Mutates the surrogate in place. Next call to `surrogate.predict_pool(...)`
    sees the augmented state.
    """
    if isinstance(surrogate, MoESurrogate):
        bundle = surrogate.bundle
        X_rf, _ = build_rf_features(
            new_feats_raw_df,
            bundle.rf_raw_feature_columns,
            bundle.rf_converted_feature_columns,
        )
        p_ps_child = float(classifier_p_ps(bundle.rf, X_rf)[0])
        is_ps = p_ps_child >= cfg.moe_threshold
        assigned = bundle.ps_expert if is_ps else bundle.nonps_expert
        _reindex_expert(assigned, new_feats_raw_df, new_z_labels, lr=cfg.learning_rate)
        return

    if isinstance(surrogate, GlobalGPRSurrogate):
        if surrogate.mode != "gpr_multitask":
            raise ValueError(
                f"KB re-conditioning only supports gpr_multitask, got mode={surrogate.mode!r}"
            )
        model_bundle = surrogate.model_bundle
        model = model_bundle["model"]
        likelihood = model_bundle["likelihood"]

        current_train_x = model.train_inputs[0]
        current_train_y = model.train_targets

        # Normalize new features via the stored global stats — same code path
        # GlobalGPRSurrogate.predict_pool uses internally.
        Xn = convert_and_normalize_features(
            new_feats_raw_df.to_numpy(dtype=np.float32),
            train=False, stats=surrogate.normalization_stats,
        )
        new_x = torch.tensor(np.asarray(Xn, dtype=np.float32), dtype=torch.float32)
        new_y = torch.tensor(np.asarray(new_z_labels).reshape(1, 2), dtype=torch.float32)

        train_x = torch.cat([current_train_x, new_x], dim=0)
        train_y = torch.cat([current_train_y, new_y], dim=0)

        model_new, likelihood_new = _retrain_model_gpr_multitask(
            cfg, model, likelihood, train_x, train_y,
        )
        # Swap in the retrained model so the surrogate sees the new state.
        model_bundle["model"] = model_new
        model_bundle["likelihood"] = likelihood_new
        return

    raise TypeError(f"Unsupported surrogate type: {type(surrogate).__name__}")


# ---------- KB inner loop (per iter × policy) ----------

def _kb_inner_loop(
    surrogate: Surrogate,
    cfg_base: ALConfig,
    train_feats_raw_df: pd.DataFrame,
    train_labels_norm_df: pd.DataFrame,
    pool_feats_raw_df: pd.DataFrame,
    pool_labels: pd.DataFrame,
    k_pick: int,
    apply_pessimism: bool,
    target_front: np.ndarray,
) -> tuple[pd.DataFrame, int]:
    """
    Kriging-believer inner loop for ONE policy at ONE iteration.

    Mirrors production `cli/child.py::run_child`: seq_id 1..k_pick, each pick
    updates the surrogate + running Pareto front before the next pick.

    Returns
    -------
    picked_real_labels : pd.DataFrame
        The real (measured) labels of the k_pick picked children. Used for
        HV computation in raw physical space.
    n_pareto_hits : int
        How many picked children lie on the target Pareto front (union of
        all completed iters).
    """
    obj1, obj2 = cfg_base.obj1, cfg_base.obj2
    front = cfg_base.front
    kind = ["max", "max"] if front == "upper" else ["min", "min"]

    # Running z-space training labels + raw features (grow as KB augments).
    running_z = train_labels_norm_df[[obj1, obj2]].to_numpy(dtype=np.float64).copy()
    running_feats = train_feats_raw_df.reset_index(drop=True).copy()

    picked_local_idxs: list[int] = []           # positional indices into the pool
    prev_feats_dfs: list[pd.DataFrame] = []     # picks earlier in this batch (for pessimism)

    n_pool = len(pool_feats_raw_df)

    for seq_id in range(1, k_pick + 1):
        # Pareto members of the running training slice, in z-space.
        _, front_indices = find_pareto_front(
            pd.DataFrame(running_z, columns=[obj1, obj2]),
            kind=kind, objectives=[obj1, obj2],
        )
        pareto_z = running_z[front_indices]
        pareto_feats_raw = running_feats.iloc[front_indices].reset_index(drop=True)

        # Epsilon-shift the front — no-op when cfg.ehvi_variant != 'epsilon'.
        pareto_input, _eps = make_epsilon_shifted_front(
            cfg=cfg_base,
            pareto_front=pareto_z,
            pareto_feats_raw_df=pareto_feats_raw,
            surrogate=surrogate,
        )

        # Remaining pool: exclude already-picked positions.
        picked_set = set(picked_local_idxs)
        remaining_pool_idxs = np.array([i for i in range(n_pool) if i not in picked_set])
        if len(remaining_pool_idxs) == 0:
            break
        remaining_feats = pool_feats_raw_df.iloc[remaining_pool_idxs].reset_index(drop=True)

        scores = score_children_ehvi(
            surrogate, remaining_feats, pareto_input, front=front,
            ref_mode=cfg_base.ref_point_mode, ref_frac=cfg_base.ref_point_frac,
        )
        best_in_remaining = int(np.argmax(scores))
        picked_local = int(remaining_pool_idxs[best_in_remaining])
        picked_local_idxs.append(picked_local)

        # KB "belief": use the surrogate's mean prediction as the child's z-labels.
        picked_feats = pool_feats_raw_df.iloc[[picked_local]].reset_index(drop=True)
        mu, cov, sig = predict_for_augmentation(surrogate, picked_feats, return_std=True)
        # mu: (1, 2), cov: (1, 2, 2), sig: (1, 2)

        # Pessimism penalty against earlier picks in this batch (iter >= start_iter, seq_id > 1).
        if apply_pessimism and len(prev_feats_dfs) > 0:
            prev_feats = pd.concat(prev_feats_dfs, ignore_index=True)
            mu_prev, cov_prev = predict_for_augmentation(surrogate, prev_feats, return_std=False)
            sign = -1.0 if front == "upper" else 1.0
            penalty = sign * overlap_batch(mu.copy(), cov[0], mu_prev, cov_prev)
            mu = mu + penalty * sig

        # Re-condition the surrogate on the picked child.
        _recondition_surrogate(surrogate, cfg_base, picked_feats, mu[0])

        # Update running training-slice state so the next seq_id's Pareto front + shift see the KB pick.
        running_z = np.vstack([running_z, mu])
        running_feats = pd.concat([running_feats, picked_feats], ignore_index=True)
        prev_feats_dfs.append(picked_feats)

    picked_real_labels = pool_labels.iloc[picked_local_idxs].reset_index(drop=True)
    n_pareto_hits = _count_pareto_hits(picked_real_labels, target_front, obj1, obj2)
    return picked_real_labels, n_pareto_hits


# ---------- orchestrator ----------

def run_retrospective(
    runs_root: Path,
    model: str,
    cfg_base: ALConfig,
    n_iters: int,
    *,
    k_pick: int | None = None,
    pessimism_start_iter: int = 6,
    log=None,
) -> dict[str, Any]:
    """
    End-to-end retrospective. Writes three artifacts under
    `cfg_base.paths.diagnostic_dir` and returns a summary dict for callers
    that want to introspect programmatically (e.g. tests).

    Parameters
    ----------
    runs_root
        SCRATCH-side root containing the completed run at
        `runs_root/<MODEL>/GENERATIONS/iteration_*/`. This is where the AL
        loop's features/labels/seqs actually live — typically
        `cfg.scratch_path`. NOT the home-side `runs/` folder, which only
        holds outputs (checkpoints, plots, logs).
    model
        Model name (e.g. "MPIPI"). Used to resolve the completed-run path.
    cfg_base
        Base ALConfig. `obj1`, `obj2`, `front`, `transform`, `ngen`,
        `ehvi_variant`, `epsilon_scale`, `ref_point_mode`, `ref_point_frac`,
        `moe_policy`, `moe_threshold`, and training hyperparams (epochs,
        patience, k_folds, learning_rate) are read. Paths + iteration +
        train_model_type are overwritten per-iter.
    n_iters
        Number of completed iters to walk (walks iter=1..n_iters).
    k_pick
        Number of children to pick per iter. Defaults to `cfg_base.ngen // 2`
        (a "half budget" retrospective — the difference between surrogates
        collapses to zero if k_pick == full batch size).
    pessimism_start_iter
        Iter at which pessimism kicks in (matches the user's production
        practice: rounds 1..5 without pessimism, 6+ with). Default 6.
    """
    log_fn = log.info if log is not None else (lambda msg: None)
    runs_root = Path(runs_root)
    diagnostic_dir = cfg_base.paths.diagnostic_dir
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    obj1, obj2, front = cfg_base.obj1, cfg_base.obj2, cfg_base.front
    if k_pick is None:
        k_pick = max(1, cfg_base.ngen // 2)

    log_fn(f"[retrospective] loading completed run from {runs_root / model}")
    all_data = load_completed_run(runs_root, model, n_iters)

    # A single ref point used everywhere so HVs are directly comparable.
    ref_pt_min = _global_ref_point(all_data.labels_df, front, obj1, obj2)

    target_front = compute_target_front(all_data.labels_df, front, obj1, obj2)
    target_hv = compute_hv_raw(
        all_data.labels_df, front=front, obj1=obj1, obj2=obj2, ref_point_min=ref_pt_min,
    )
    log_fn(f"[retrospective] target HV (union of all iters) = {target_hv:.4f}")

    # Running set of picked children under each policy — starts as the seed pool
    # (iter 0), which every policy sees identically.
    seed = all_data.labels_df[all_data.labels_df["generation"] == 0].copy()
    picks: dict[str, pd.DataFrame] = {
        "moe_soft": seed.copy(),
        "moe_hard": seed.copy(),
        "global":   seed.copy(),
    }
    actual_running = seed.copy()

    hv_traj: dict[str, list[float]] = {name: [] for name in list(picks) + ["actual"]}
    hv_traj_iters: list[int] = []

    summary_rows: list[dict[str, Any]] = []

    for M in range(1, n_iters + 1):
        log_fn(f"[retrospective] iter {M}: training surrogates on generations < {M}")
        train_feats, train_labels, _train_seqs = all_data.training_slice_before(M)
        pool_feats, pool_labels, _pool_seqs = all_data.proposal_pool_at(M)

        if len(pool_labels) == 0:
            log_fn(f"[retrospective] iter {M}: no proposal children found, skipping.")
            continue

        # Train fresh surrogates for this iter on the pre-iter-M data slice.
        # Read the shared labels_norm_csv BEFORE the tempdir goes away — the KB
        # loop needs the full training-slice z-labels to grow.
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)

            # MoE
            cfg_moe = _make_iter_cfg(tempdir / "moe", cfg_base, iteration=M - 1, train_model_type="moe")
            _write_training_slice(cfg_moe, train_feats, train_labels)
            try:
                moe_surs = _build_moe_surrogates(cfg_moe)
                moe_labels_norm_df = pd.read_csv(cfg_moe.paths.labels_norm_csv)
            except Exception as e:
                log_fn(f"[retrospective] iter {M}: MoE training failed ({e!r}), skipping MoE this iter.")
                moe_surs = {}
                moe_labels_norm_df = None

            # Global
            cfg_global = _make_iter_cfg(tempdir / "global", cfg_base, iteration=M - 1, train_model_type="gpr_multitask")
            _write_training_slice(cfg_global, train_feats, train_labels)
            global_sur = _build_global_surrogate(cfg_global)
            global_labels_norm_df = pd.read_csv(cfg_global.paths.labels_norm_csv)

        # KB inner loop per policy — mirrors production run_child's seq_id loop.
        row: dict[str, Any] = {"iter": M, "n_children": int(len(pool_labels)), "n_picked": int(min(k_pick, len(pool_labels)))}
        apply_pessimism = (M >= pessimism_start_iter)

        for name, sur, labels_norm_df in [
            ("moe_soft",  moe_surs.get("moe_soft"),  moe_labels_norm_df),
            ("moe_hard",  moe_surs.get("moe_hard"),  moe_labels_norm_df),
            ("global",    global_sur,                global_labels_norm_df),
        ]:
            if sur is None or labels_norm_df is None:
                row[f"n_pareto_members_hit_{name}"] = 0
                hv_traj[name].append(hv_traj[name][-1] if hv_traj[name] else 0.0)
                continue
            picked, n_hits = _kb_inner_loop(
                surrogate=sur,
                cfg_base=cfg_base,
                train_feats_raw_df=train_feats,
                train_labels_norm_df=labels_norm_df,
                pool_feats_raw_df=pool_feats,
                pool_labels=pool_labels,
                k_pick=row["n_picked"],
                apply_pessimism=apply_pessimism,
                target_front=target_front,
            )
            picks[name] = pd.concat([picks[name], picked], ignore_index=True)
            hv_traj[name].append(compute_hv_raw(
                picks[name], front=front, obj1=obj1, obj2=obj2, ref_point_min=ref_pt_min,
            ))
            row[f"n_pareto_members_hit_{name}"] = int(n_hits)

        # Baseline: HV if you kept EVERY iter-M child (what the real AL loop did).
        actual_running = pd.concat([actual_running, pool_labels], ignore_index=True)
        hv_traj["actual"].append(compute_hv_raw(
            actual_running, front=front, obj1=obj1, obj2=obj2, ref_point_min=ref_pt_min,
        ))
        hv_traj_iters.append(M)

        for name in ["moe_soft", "moe_hard", "global", "actual"]:
            row[f"hv_{name}"] = hv_traj[name][-1]
        summary_rows.append(row)
        log_fn(f"[retrospective] iter {M}: "
                f"HV actual={hv_traj['actual'][-1]:.4f} "
                f"moe_soft={hv_traj['moe_soft'][-1]:.4f} "
                f"moe_hard={hv_traj['moe_hard'][-1]:.4f} "
                f"global={hv_traj['global'][-1]:.4f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(diagnostic_dir / "retrospective_summary.csv", index=False)

    trajectory = {
        "iters":                hv_traj_iters,
        "target_hv":            target_hv,
        "target_front":         target_front.tolist(),
        "hv_actual":            hv_traj["actual"],
        "hv_moe_soft":          hv_traj["moe_soft"],
        "hv_moe_hard":          hv_traj["moe_hard"],
        "hv_global":            hv_traj["global"],
        "k_pick":               k_pick,
        "pessimism_start_iter": pessimism_start_iter,
        "ehvi_variant":         cfg_base.ehvi_variant,
        "epsilon_scale":        cfg_base.epsilon_scale,
        "ref_point_mode":       cfg_base.ref_point_mode,
        "ref_point_frac":       cfg_base.ref_point_frac,
        "ref_point_min":        ref_pt_min.tolist(),
        "front":                front,
        "obj1":                 obj1,
        "obj2":                 obj2,
        "rounds_to_95pct": {
            name: _rounds_to_hv(hv_traj[name], hv_traj_iters, 0.95 * target_hv)
            for name in ("actual", "moe_soft", "moe_hard", "global")
        },
    }
    with open(diagnostic_dir / "retrospective_trajectory.json", "w") as f:
        json.dump(trajectory, f, indent=2)

    log_fn(f"[retrospective] wrote {diagnostic_dir / 'retrospective_summary.csv'}")
    log_fn(f"[retrospective] wrote {diagnostic_dir / 'retrospective_trajectory.json'}")

    return {
        "summary_df":  summary_df,
        "trajectory":  trajectory,
        "target_hv":   target_hv,
        "target_front": target_front,
    }


# ---------- small helpers ----------

def _count_pareto_hits(picked: pd.DataFrame, target_front: np.ndarray,
                        obj1: str, obj2: str, atol: float = 1e-8) -> int:
    """How many rows in `picked` correspond to points on the target Pareto front."""
    if len(picked) == 0 or len(target_front) == 0:
        return 0
    pts = picked[[obj1, obj2]].to_numpy()
    hits = 0
    for row in pts:
        if np.any(np.all(np.isclose(target_front, row, atol=atol), axis=1)):
            hits += 1
    return hits


def _rounds_to_hv(hv_traj: list[float], iters: list[int], threshold: float) -> int | None:
    """First iter where cumulative HV crosses `threshold`; None if never."""
    for h, it in zip(hv_traj, iters):
        if h >= threshold:
            return it
    return None
