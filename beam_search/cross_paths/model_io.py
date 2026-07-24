# cross_paths/model_io.py
#
# Row 8 beam-surrogate-cleanup: the file is a thin, drift-safe wrapper around
# `al_pipeline.surrogates.load_surrogate` + the ported numba featurizer.
#
# Removed vs. the pre-refactor version:
#   * `_fit_label_scalers`  — refit `PowerTransformer` at load time on the
#     labeled data. Discarded the persisted scalers the AL surrogate used;
#     drift-prone. Bundles now expose the scalers baked into `MoEBundle`
#     (via each `GPRExpert.label_scaler1/2`), shared under
#     `label_scaler_scope='all'`.
#   * `standard_normalize_features_vec`  — reimplemented feature normalization
#     with hardcoded column indices and a JSON side-channel. Normalization
#     now happens inside `Surrogate.predict_design`; the caller passes raw
#     featurizer output DataFrames.
#   * `sequence_featurizer_numba` / `sequence_featurizer` imports off
#     `PYTHONPATH`. Ported to
#     `al_pipeline.featurization.sequence_featurizer_numba` and imported by
#     dotted path here.
#
# The public surface preserved for beam_search.py / run_beams_mpi.py:
#   * `load_all_models(paths, db_dir) -> {model: BeamBundle}` (kept for API
#     compat; a single-model wrapper around `load_beam_bundle`).
#   * `predict_labels_for_sequences(bundle, seqs, ..., feat_threads=1)`
#     returns z-space means (B, 2) matching the pre-refactor signature so
#     `predict_candidate_frames` in beam_search.py is untouched by Row 8.
#     Row 9 (`feat/beam-policy`) will collapse this call into the policy
#     layer alongside the physical + gate outputs from `predict_design`.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from al_pipeline.core.config import ALConfig
from al_pipeline.core.paths import ALPaths
from al_pipeline.featurization.sequence_featurizer_numba import (
    SequenceFeaturizer as SequenceFeaturizerNumba,
)
from al_pipeline.surrogates import DesignPrediction, Surrogate, load_surrogate
from al_pipeline.surrogates.moe import MoESurrogate


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

@dataclass
class BeamBundle:
    """Everything the beam-search MPI dispatch needs for one model.

    ``surrogate`` handles featurization + normalization + prediction end to
    end via ``predict_design``. The other fields carry the labeled iter-N
    training-set arrays the beam uses for start selection (III.2) and the
    endpoints-CSV bookkeeping (III.8): both physical objective values
    (``labels_exp_density``, ``labels_diff``) and the regime label
    (``density``).

    ``label_scalers`` is a ``(scaler1, scaler2)`` pair matching the shared
    scalers from ``MoEBundle`` (both experts hold identical instances under
    ``label_scaler_scope='all'``). Row 8's `predict_labels_for_sequences`
    doesn't use them, but ``beam_search.beam_search_paths`` still calls
    ``label_scalers[i].transform`` to place the start sequence in z-space —
    keep them exposed until Row 9 replaces that path.
    """
    model_name: str
    surrogate: Surrogate
    sequences: List[str]
    features: pd.DataFrame                    # raw featurizer output (N, 29)
    labels_exp_density: np.ndarray            # (N,) physical
    labels_diff: np.ndarray                   # (N,) physical
    labels_density: np.ndarray | None         # (N,) physical or None if column absent
    start_regime: np.ndarray | None           # (N,) bool: density > 0
    label_scaler1: Any                        # inherits from persisted MoE / GPRExpert
    label_scaler2: Any
    featurizer: SequenceFeaturizerNumba

    @property
    def labels(self) -> np.ndarray:
        """(N, 2) physical labels stacked as (exp_density, diff) for legacy callers."""
        return np.stack([self.labels_exp_density, self.labels_diff], axis=-1)

    @property
    def label_scalers(self) -> tuple:
        """(scaler1, scaler2) tuple — pre-refactor compat for beam_search_paths."""
        return (self.label_scaler1, self.label_scaler2)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _cfg_from_paths(paths: ALPaths, *, db_path: Path) -> ALConfig:
    """Build a minimal `ALConfig` from an `ALPaths` for surrogate loading.

    Beam prepare_endpoints hands us an `ALPaths` (the pre-Row-8 API). The
    surrogate loader wants an `ALConfig` — so lift the fields the loader
    reads: base_path/scratch_path, iteration, model, front,
    ehvi/exploration/transform, train_model_type='moe', obj1/obj2/aux1_obj1.

    ``ALConfig.paths`` is a derived @property, so we can't hand it in
    directly; we pass the underlying ``base_path`` / ``scratch_path`` and
    let the config reconstruct the equivalent ``ALPaths``.
    """
    return ALConfig(
        model=paths.model,
        iteration=paths.iteration,
        front=paths.front,
        ehvi_variant=paths.ehvi_variant,
        exploration_strategy=paths.exploration_strategy,
        transform=paths.transform,
        mc_ehvi=paths.mc_ehvi,
        base_path=Path(paths.base_path),
        scratch_path=Path(paths.scratch_path),
        db_path=Path(db_path),
        train_model_type="moe",
        obj1="exp_density",
        obj2="diff",
        aux1_obj1="density",
    )


def _scaler_pair_from_surrogate(surrogate: Surrogate) -> tuple:
    """Extract ``(label_scaler1, label_scaler2)`` from an MoE surrogate.

    Both experts hold identical instances under ``label_scaler_scope='all'``
    — read them off the PS expert. Global GPR surrogates don't persist
    scalers (see `GlobalGPRSurrogate.predict_design` docstring), so beam-side
    consumers using ``--policy global`` in a future iteration will need a
    separate scaler-load path.
    """
    if not isinstance(surrogate, MoESurrogate):
        raise TypeError(
            "Beam bundle currently only supports MoE surrogates for scaler "
            "extraction. Global-GPR beam runs need scaler persistence in "
            "kfold_training first (deferred, see plan §III.6)."
        )
    ps_expert = surrogate.bundle.ps_expert
    return (ps_expert.label_scaler1, ps_expert.label_scaler2)


def load_beam_bundle(
    paths: ALPaths,
    db_dir: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> BeamBundle:
    """Load the full beam-side bundle for one model at ``paths.iteration``.

    Pulls the surrogate through `al_pipeline.surrogates.load_surrogate`,
    loads the labeled features + both objective labels + the ``density``
    regime column (per III.8 of the beam plan), and instantiates the numba
    featurizer against the FF database at ``db_dir``.

    Parameters
    ----------
    paths : ALPaths
        Fully-populated. Provides features_csv, labels_csv, seq_gen_txt.
    db_dir : str or Path
        Directory containing ``ff_db.py`` and the model parameter files.
    device : torch.device or str
        Torch device for the surrogate's GP tensors. Default ``"cpu"``.
        Pass ``"cuda"`` (or ``"cuda:0"``) on GPU nodes to run predict_design
        on GPU; the CPU↔GPU round-trip happens once per predict_design call
        (small, microseconds).
    """
    db_dir = Path(db_dir)
    cfg = _cfg_from_paths(paths, db_path=db_dir)

    # Existence check per §IV.pre-diagnostic-verification.
    for name, p in {
        "features_csv": paths.features_csv,
        "labels_csv":   paths.labels_csv,
        "seq_gen_txt":  paths.seq_gen_txt,
        "moe_ps":       paths.moe_ps_chkpt(temp=False),
        "moe_nonps":    paths.moe_nonps_chkpt(temp=False),
        "moe_rf":       paths.moe_rf_bundle(temp=False),
    }.items():
        if not Path(p).exists():
            raise FileNotFoundError(f"[{paths.model}] Missing {name}: {p}")

    surrogate = load_surrogate(cfg, temp=False, device=device)

    # Load features + all three physical label columns in one shot. Load
    # `density` alongside `exp_density` and `diff` per III.8 — the regime
    # label is `density > 0`, the quantile axis is `exp_density`, and both
    # need to be recorded in the endpoints CSV.
    labels_df = pd.read_csv(paths.labels_csv)
    features_df = pd.read_csv(paths.features_csv)

    for col in ("exp_density", "diff"):
        if col not in labels_df.columns:
            raise KeyError(f"[{paths.model}] labels_csv missing required column {col!r}")
    # `density` is optional at load time but required for regime split;
    # surface a clear error rather than silently defaulting to `exp_density`.
    density_col = "density"
    if density_col not in labels_df.columns:
        raise KeyError(
            f"[{paths.model}] labels_csv missing 'density' column — needed "
            "for regime split (III.8). Was moe_regime_oof.py run against a "
            "labels CSV that has this column? See plan §I.2-B."
        )

    labels_df_clean = labels_df.dropna(subset=["exp_density", "diff", density_col])
    features_clean = features_df.loc[labels_df_clean.index].reset_index(drop=True)
    labels_clean = labels_df_clean.reset_index(drop=True)

    labels_exp_density = labels_clean["exp_density"].to_numpy(dtype=np.float64)
    labels_diff = labels_clean["diff"].to_numpy(dtype=np.float64)
    labels_density = labels_clean[density_col].to_numpy(dtype=np.float64)
    start_regime = labels_density > 0.0

    # Sequences — one per feature row. seq_gen_txt is the AL training
    # sequences file; slice to the clean-index subset in case any labels were
    # dropped above.
    with open(paths.seq_gen_txt, "r") as f:
        all_seqs = [ln.strip() for ln in f if ln.strip()]
    if len(all_seqs) != len(labels_df):
        raise ValueError(
            f"[{paths.model}] sequence count mismatch: seq_gen_txt has "
            f"{len(all_seqs)} rows but labels_csv has {len(labels_df)}"
        )
    sequences_clean = [all_seqs[i] for i in labels_df_clean.index]

    label_scaler1, label_scaler2 = _scaler_pair_from_surrogate(surrogate)
    featurizer = SequenceFeaturizerNumba(paths.model.lower(), str(db_dir))

    return BeamBundle(
        model_name=paths.model,
        surrogate=surrogate,
        sequences=sequences_clean,
        features=features_clean,
        labels_exp_density=labels_exp_density,
        labels_diff=labels_diff,
        labels_density=labels_density,
        start_regime=start_regime,
        label_scaler1=label_scaler1,
        label_scaler2=label_scaler2,
        featurizer=featurizer,
    )


def load_all_models(
    paths: ALPaths,
    db_dir: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> Dict[str, BeamBundle]:
    """API-compat wrapper: one-model bundle keyed by ``paths.model``.

    Beam callers that predate the surrogate refactor still iterate
    ``bundles[model_name]``. Multi-model dispatch is not currently a beam
    concern; the caller loops over models separately. ``device`` forwards
    to `load_beam_bundle` for GPU-enabled beam runs.
    """
    return {paths.model: load_beam_bundle(paths, db_dir, device=device)}


# ---------------------------------------------------------------------------
# Batched prediction
# ---------------------------------------------------------------------------

def predict_labels_for_sequences(
    bundle: BeamBundle,
    sequences: List[str],
    return_std: bool = False,
    batch_size: int = 4096,
    feat_threads: int = 1,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Batched predict in z-space via `Surrogate.predict_design`.

    Preserves the pre-refactor return shape so `beam_search.beam_search_paths`
    (which was written against z-space z outputs and does its own physical
    inversion via ``bundle.label_scalers``) works unchanged. Row 9 will
    replace the caller with a policy-aware path that consumes
    `predict_design` directly (per-expert + physical + gate).

    Parameters
    ----------
    bundle : BeamBundle
        From `load_beam_bundle`.
    sequences : list[str]
        Candidates to predict.
    return_std : bool
        If True, return ``(mu, sd)`` in z-space rather than just ``mu``.
    batch_size : int
        Passed to the featurizer / surrogate for chunking. Currently the
        surrogate reads all at once; kept for API compat.
    feat_threads : int
        Passed to the numba featurizer.
    """
    del batch_size  # surrogate batches internally; kept for API compat
    if not sequences:
        empty = np.zeros((0, 2), dtype=np.float64)
        return (empty, empty) if return_std else empty

    feat_threads_eff = 1 if len(sequences) < 64 else int(feat_threads)
    X = bundle.featurizer.featurize_many_fast(
        sequences, feat_threads_eff, as_df=True,
    )
    pred: DesignPrediction = bundle.surrogate.predict_design(X)
    z_mean = np.asarray(pred.z_mean, dtype=np.float64)
    if return_std:
        z_std = np.asarray(pred.z_std, dtype=np.float64)
        return z_mean, z_std
    return z_mean
