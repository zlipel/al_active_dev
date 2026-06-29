# cross_paths/model_io.py
#
# Merged from cross_paths/io.py and cross_paths/io_test.py.
# Canonical implementation: io_test.py (correct local path, separate if/else branches).
#
# Changes vs either source file:
#   - All path resolution uses ALPaths instances (no hardcoded strings, no os.getcwd heuristic)
#   - GPR checkpoint: tries al_pipeline naming first, falls back to legacy (no _{front}) with warning
#   - Normalization stats: handles both old {std_normal_dict, maxS} and new {means, stds, max_S} formats
#   - standard_normalize_features (scalar version, unused) removed
#   - Dead commented-out code blocks removed
#   - Imports for gpr_model, data_preprocessing_gpr, sequence_featurizer* come from al_pipeline

from __future__ import annotations
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gpytorch
import numpy as np
import torch
from sklearn.preprocessing import PowerTransformer
from time import time

from al_pipeline.core.paths import ALPaths
from al_pipeline.training.ml_models import MultitaskGPRegressionModel
from al_pipeline.data_prep.data_loading import load_dataset

import sequence_featurizer_numba as sff   # fast featurizer (must be on PYTHONPATH)
import sequence_featurizer as sf           # slower / reference featurizer


# ---------------------------------------------------------------------------
# Compatibility shims for legacy on-disk formats
# ---------------------------------------------------------------------------

def _load_norm_stats(path: Path) -> dict:
    """
    Load normalization_stats.json, handling both on-disk formats:

    al_pipeline format (new):
        {"means": {feat: float}, "stds": {feat: float}, "min_L": ..., "max_L": ..., "max_S": ...}

    Legacy format (old calculate_normalization_stats.py):
        {"std_normal_dict": {feat: [mean, std]}, "min_L": ..., "max_L": ..., "maxS": ...}
    """
    with open(path) as f:
        raw = json.load(f)
    if "means" in raw:
        return raw  # al_pipeline format — pass through
    elif "std_normal_dict" in raw:
        warnings.warn(
            "Legacy normalization_stats.json format detected. "
            "Regenerate by re-running training with al_pipeline.",
            DeprecationWarning,
            stacklevel=2,
        )
        snd = raw["std_normal_dict"]
        return {
            "means": {k: v[0] for k, v in snd.items()},
            "stds":  {k: v[1] for k, v in snd.items()},
            "min_L": raw["min_L"],
            "max_L": raw["max_L"],
            "max_S": raw["maxS"],
        }
    else:
        raise ValueError(f"Unrecognized normalization_stats.json schema in {path}")


def _resolve_gpr_checkpoint(paths: ALPaths) -> Path:
    """
    Try the al_pipeline checkpoint name first (includes _{front}).
    Fall back to the legacy name (no _{front} suffix) with a deprecation warning.
    Raise FileNotFoundError if neither exists.
    """
    new_path = paths.gpr_multitask_chkpt(temp=False)
    if new_path.exists():
        return new_path

    # Legacy pattern: GPR_iter{N}_{ehvi_variant}_{exploration_strategy}_{transform}.pt
    old_name = (
        f"GPR_iter{paths.iteration}_"
        f"{paths.ehvi_variant}_"
        f"{paths.exploration_strategy}_"
        f"{paths.transform}.pt"
    )
    old_path = paths.models_dir / old_name
    if old_path.exists():
        warnings.warn(
            f"Legacy checkpoint name detected: {old_path.name}. "
            "Retrain with al_pipeline to use new naming convention (includes _{front}).",
            DeprecationWarning,
            stacklevel=2,
        )
        return old_path

    raise FileNotFoundError(
        f"GPR checkpoint not found.\n  Tried: {new_path}\n  Tried: {old_path}"
    )


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ModelResources:
    model_name: str
    model: gpytorch.models.ExactGP
    likelihood: gpytorch.likelihoods.Likelihood
    label_scalers: Tuple[PowerTransformer, PowerTransformer]  # (exp_density, diff)
    sequences: List[str]
    features: np.ndarray
    labels: np.ndarray
    normalization_stats: dict
    featurizer_fast: object
    featurizer_slow: object
    device: torch.device
    feature_dim: int


# ---------------------------------------------------------------------------
# Label scalers
# ---------------------------------------------------------------------------

def _fit_label_scalers(
    labels_exp: torch.Tensor,
    labels_diff: torch.Tensor,
) -> Tuple[Tuple[PowerTransformer, PowerTransformer], torch.Tensor, torch.Tensor]:
    s1 = PowerTransformer(method="yeo-johnson", standardize=True)
    s2 = PowerTransformer(method="yeo-johnson", standardize=True)
    y1 = torch.tensor(
        s1.fit_transform(labels_exp.view(-1, 1)), dtype=torch.float32
    ).flatten()
    y2 = torch.tensor(
        s2.fit_transform(labels_diff.view(-1, 1)), dtype=torch.float32
    ).flatten()
    return (s1, s2), y1, y2


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_bundle(
    paths: ALPaths,
    db_dir: str | Path,
    device: Optional[torch.device] = None,
) -> ModelResources:
    """
    Load a single model bundle (weights, likelihood, scalers, featurizers,
    sequences, normalization stats) using an ALPaths instance for all path
    resolution.

    Parameters
    ----------
    paths : ALPaths
        Fully-populated ALPaths instance (model, iteration, front, etc.).
        Provides all file locations: features_csv, labels_csv, seq_gen_txt,
        norm_stats, models_dir.
    db_dir : str or Path
        Path to the sequence feature databases directory.
    device : torch.device, optional
        Inference device. Defaults to CPU.
    """
    db_dir = Path(db_dir)

    # Resolve GPR checkpoint (al_pipeline naming, with legacy fallback)
    ckpt_path = _resolve_gpr_checkpoint(paths)

    # Validate required paths
    required = {
        "features_csv": paths.features_csv,
        "labels_csv":   paths.labels_csv,
        "seq_gen_txt":  paths.seq_gen_txt,
        "norm_stats":   paths.norm_stats,
    }
    for name, p in required.items():
        if not p.exists():
            raise FileNotFoundError(f"[{paths.model}] Missing {name}: {p}")

    # Load features + labels
    feats_np, exp_density_np = load_dataset(
        str(paths.features_csv), str(paths.labels_csv), label_column="exp_density"
    )
    _, diff_np = load_dataset(
        str(paths.features_csv), str(paths.labels_csv), label_column="diff"
    )

    feats = torch.tensor(feats_np, dtype=torch.float32)
    y1 = torch.tensor(exp_density_np, dtype=torch.float32).flatten()
    y2 = torch.tensor(diff_np, dtype=torch.float32).flatten()

    # Fit label scalers on the same data slice
    (scaler1, scaler2), y1_scaled, y2_scaled = _fit_label_scalers(y1, y2)
    labels_scaled = torch.stack([y1_scaled, y2_scaled], dim=-1)  # (N, 2)

    # Load GP model
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    gp = MultitaskGPRegressionModel(feats, labels_scaled, likelihood, num_tasks=2)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    gp.load_state_dict(ckpt["model"])
    gp.eval()
    likelihood.eval()

    # Device placement
    if device is None:
        device = torch.device("cpu")
    gp.to(device)
    likelihood.to(device)
    feats = feats.to(device)

    # Sequences + normalization stats + featurizers
    with open(paths.seq_gen_txt, "r") as f:
        seqs = [ln.strip() for ln in f if ln.strip()]

    norm_stats = _load_norm_stats(paths.norm_stats)
    feat_fast = sff.SequenceFeaturizer(paths.model.lower(), str(db_dir))
    feat_slow = sf.SequenceFeaturizer(paths.model.lower(), str(db_dir))

    return ModelResources(
        model_name=paths.model,
        model=gp,
        likelihood=likelihood,
        label_scalers=(scaler1, scaler2),
        sequences=seqs,
        features=feats_np,
        labels=np.stack([exp_density_np, diff_np], axis=-1),
        normalization_stats=norm_stats,
        featurizer_fast=feat_fast,
        featurizer_slow=feat_slow,
        device=device,
        feature_dim=feats.shape[1],
    )


def load_all_models(
    paths: ALPaths,
    db_dir: str | Path,
) -> Dict[str, ModelResources]:
    """
    Load a single model bundle as a dict keyed by model name.
    (ALPaths encodes one model; call separately for each model if needed.)
    """
    return {paths.model: load_model_bundle(paths, db_dir)}


# ---------------------------------------------------------------------------
# Feature normalization
# ---------------------------------------------------------------------------

def standard_normalize_features_vec(X: np.ndarray, normalization_stats: dict) -> np.ndarray:
    """
    Normalize a feature array in-place and return it.

    X: shape (29,) or (N, 29)
    Applies:
      - divide count features and selected scalar features by sequence length
      - min-max normalize length
      - z-score normalize SCD, SHD, |net charge|, sum lambda, beads(+/-), mol wt
      - scale shannon entropy: S / max_S - 1
    """
    means_d = normalization_stats["means"]
    stds_d  = normalization_stats["stds"]
    min_L   = normalization_stats["min_L"]
    max_L   = normalization_stats["max_L"]
    max_S   = normalization_stats["max_S"]

    X = np.asarray(X)
    denom = max_L - min_L

    feat_names = ["SCD", "SHD", "|net charge|", "sum lambda", "beads(+)", "beads(-)", "mol wt"]
    idxs  = np.array([21, 22, 23, 24, 25, 26, 28], dtype=int)
    means = np.array([means_d[f] for f in feat_names], dtype=X.dtype)
    stds  = np.array([stds_d[f]  for f in feat_names], dtype=X.dtype)

    if X.ndim == 1:
        L    = X[20]
        invL = 1.0 / L

        X[:20] *= invL
        X[23]  *= invL
        X[24]  *= invL
        X[25]  *= invL
        X[26]  *= invL
        X[28]  *= invL

        X[20]   = (X[20] - min_L) / denom
        X[27]   = X[27] / max_S - 1.0
        X[idxs] = (X[idxs] - means) / stds
        return X

    # 2D case: (N, 29)
    L    = X[:, 20]
    invL = 1.0 / L

    X[:, :20] *= invL[:, None]
    X[:, 23]  *= invL
    X[:, 24]  *= invL
    X[:, 25]  *= invL
    X[:, 26]  *= invL
    X[:, 28]  *= invL

    X[:, 20]   = (X[:, 20] - min_L) / denom
    X[:, 27]   = X[:, 27] / max_S - 1.0
    X[:, idxs] = (X[:, idxs] - means[None, :]) / stds[None, :]
    return X


# ---------------------------------------------------------------------------
# Batched prediction
# ---------------------------------------------------------------------------

def predict_labels_for_sequences(
    bundle: ModelResources,
    sequences: List[str],
    return_std: bool = False,
    batch_size: int = 4096,
    feat_threads: int = 1,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """
    Featurize → normalize → GP posterior in one call.

    Returns scaled-prediction means (and optionally stds) in the same scaled
    space the model was trained on. To get physical units, apply
    bundle.label_scalers[i].inverse_transform() outside.
    """
    t0 = time()

    # Featurize
    feat_threads_eff = 1 if len(sequences) < 64 else feat_threads
    X = bundle.featurizer_fast.featurize_many_fast(sequences, feat_threads_eff, as_df=False)

    t1 = time()

    # Normalize
    X = standard_normalize_features_vec(X, bundle.normalization_stats)

    t2 = time()

    # To tensor
    Xt = torch.tensor(X, dtype=torch.float32, device=bundle.device)

    t3 = time()

    # Predict in batches
    means = []
    bundle.model.eval()
    bundle.likelihood.eval()

    if return_std:
        stds = []
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(0, Xt.shape[0], batch_size):
                xb   = Xt[i : i + batch_size]
                post = bundle.model(xb)
                m    = post.mean.detach().cpu().numpy()       # (B, 2)
                v    = post.variance.detach().cpu().numpy()   # (B, 2)
                means.append(m)
                stds.append(np.sqrt(v))

        mu = np.vstack(means)
        sd = np.vstack(stds)
        return mu, sd

    else:
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(0, Xt.shape[0], batch_size):
                xb   = Xt[i : i + batch_size]
                post = bundle.model(xb)
                m    = post.mean.detach().cpu().numpy()       # (B, 2)
                means.append(m)

        mu = np.vstack(means)
        return mu
