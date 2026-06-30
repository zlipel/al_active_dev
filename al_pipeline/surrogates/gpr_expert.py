"""
One trained multitask GPR ("expert") with the preprocessing it needs to use.

The MoE surrogate holds three of these — global / PS / nonPS — and blends their
outputs by a gate. Beam search holds one or more directly for its own scoring.

A `GPRExpert` bundles:
  - the live `MultitaskGPRegressionModel` + likelihood (in eval mode after load)
  - the per-expert feature normalizer stats (each expert is fit on its own row
    subset, so each has its own mean/std/range per feature)
  - the two label scalers (one per objective; shared across experts when
    `label_scaler_scope='all'`, which is the only mode the AL MoE supports)
  - the label transform (yeoj / log / none) used before scaling

Construction is via `train(...)` on raw rows, or `from_checkpoint(...)` to
reload a saved expert. `predict(...)` returns z-space (the GP output space)
and physical-space predictions for both objectives.

Provenance fields (`label_scaler_scope`, `model_name`, `iteration`, `regime`)
are populated when loaded from a checkpoint; the `MoEBundle` validates that
the three experts agree on them.
"""
from __future__ import annotations

import os
from typing import Any

import gpytorch
import numpy as np
import pandas as pd
import torch

from al_pipeline.data_prep.data_loading import (
    apply_feature_normalizer,
    convert_features,
    fit_feature_normalizer,
)
from al_pipeline.training.ml_models import MultitaskGPRegressionModel
from al_pipeline.training.trainers import MultitaskGPRTrainer


def _prepare_label_array(labels_df: pd.DataFrame, label_columns: list[str], transform: str) -> np.ndarray:
    """
    Apply the pre-scaler label transform (yeoj/log/none) and return a (N, 2) array.

    Matches the MCSC `prepare_label_array` semantics so existing checkpoints
    round-trip identically. Lives here as a private helper for now; will be
    lifted into a shared `data_prep/label_transforms.py` when feat/moe-training
    consolidates the training pipeline.
    """
    y = labels_df[list(label_columns)].to_numpy(dtype=np.float64).copy()
    if transform == "log":
        # The log transform is applied only to the second objective (e.g. diff);
        # the first objective is left for the scaler to standardize. Matches the
        # existing global-GPR pipeline at kfold_training.py.
        y[:, 1] = np.log(y[:, 1] + 1e-8)
    elif transform in {"yeoj", "none"}:
        pass
    else:
        raise ValueError(f"Unknown transform={transform!r}")
    return y


class GPRExpert:
    """A trained multitask GPR + the preprocessing required to predict with it."""

    def __init__(
        self,
        model: MultitaskGPRegressionModel,
        likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
        feature_normalizer_stats: dict,
        label_scaler1: Any,
        label_scaler2: Any,
        transform: str,
        label_columns: list[str],
        feature_columns: list[str],
        label_scaler_scope: str | None = None,
        model_name: str | None = None,
        iteration: int | None = None,
        regime: str | None = None,
    ):
        self.model = model
        self.likelihood = likelihood
        self.feature_normalizer_stats = feature_normalizer_stats
        self.label_scaler1 = label_scaler1
        self.label_scaler2 = label_scaler2
        self.transform = transform
        self.label_columns = list(label_columns)
        self.feature_columns = list(feature_columns)
        # Provenance: populated when loaded from a checkpoint, None for freshly trained.
        self.label_scaler_scope = label_scaler_scope
        self.model_name = model_name
        self.iteration = iteration
        self.regime = regime

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    @classmethod
    def train(
        cls,
        features_raw_df: pd.DataFrame,
        labels_raw_df: pd.DataFrame,
        label_columns: list[str],
        transform: str,
        scaler1: Any,
        scaler2: Any,
        feature_columns: list[str],
        lr: float = 0.1,
        epochs: int = 1000,
        patience: int = 5,
    ) -> "GPRExpert":
        """
        Train one expert on the supplied raw rows.

        The feature normalizer is fit on THESE rows (per-expert). Label scalers
        are supplied by the caller — typically fit on the full training set
        before training the experts, under `label_scaler_scope='all'`.

        For test purposes the caller can pass tiny synthetic data and a low
        epoch count; in production the same call is made by the (future) MoE
        training pipeline.
        """
        feats_raw = features_raw_df[list(feature_columns)]
        feat_conv = convert_features(feats_raw)
        feat_stats = fit_feature_normalizer(feat_conv)
        feat_norm = apply_feature_normalizer(feat_conv, feat_stats)

        lab_prepared = _prepare_label_array(labels_raw_df, label_columns, transform)
        lab_scaled = lab_prepared.copy()
        lab_scaled[:, 0] = scaler1.transform(lab_prepared[:, [0]]).ravel()
        lab_scaled[:, 1] = scaler2.transform(lab_prepared[:, [1]]).ravel()

        feats_t = torch.tensor(feat_norm.to_numpy(), dtype=torch.float32)
        labels_t = torch.tensor(lab_scaled, dtype=torch.float32)

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
        model = MultitaskGPRegressionModel(feats_t, labels_t, likelihood, num_tasks=2)
        trainer = MultitaskGPRTrainer(
            model, likelihood, learning_rate=lr, epochs=epochs, patience=patience,
        )
        trainer.train((feats_t, labels_t), None, early_stop=True)
        model.eval()
        likelihood.eval()

        return cls(
            model, likelihood, feat_stats, scaler1, scaler2,
            transform, label_columns, list(feature_columns),
        )

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def _feature_tensor(self, feats_raw_df: pd.DataFrame) -> torch.Tensor:
        """Convert + normalize raw features using THIS expert's stats."""
        missing = [c for c in self.feature_columns if c not in feats_raw_df.columns]
        if missing:
            raise ValueError(f"Features missing columns required by expert: {missing}")
        feats_raw = feats_raw_df[self.feature_columns]
        feat_conv = convert_features(feats_raw)
        feat_norm = apply_feature_normalizer(feat_conv, self.feature_normalizer_stats)
        return torch.tensor(feat_norm.to_numpy(), dtype=torch.float32)

    def posterior(self, feats_raw_df: pd.DataFrame) -> gpytorch.distributions.MultitaskMultivariateNormal:
        """
        Return the raw gpytorch joint posterior over (obj1, obj2) for these rows.

        The MoE surrogate uses this to sample for MC-EHVI; the `predict` method
        below builds on it for z-space / physical-space mean+std summaries.
        """
        x = self._feature_tensor(feats_raw_df)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            return self.likelihood(self.model(x))

    def predict(self, feats_raw_df: pd.DataFrame) -> dict[str, np.ndarray]:
        """
        Predict z-space (GP output) values for both tasks.

        z-space is what the AL acquisition consumes (Pareto front, EHVI, MC
        sampling all live in normalized objective space). Physical-space
        outputs are NOT returned — diagnostic consumers that want them should
        call `self.label_scaler1.inverse_transform(out["exp_density_z_mean"])`
        and apply the diff-side inverse transform themselves.

        Returns
        -------
        dict
            Keys per objective:
              exp_density_z_mean, exp_density_z_var, exp_density_std_norm,
              diff_z_mean,        diff_z_var,        diff_std_norm
            All shape (N,). Variances are clipped to >= 0.
        """
        with torch.no_grad():
            dist = self.posterior(feats_raw_df)
            mean_z = dist.mean.detach().cpu().numpy()
            var_z = dist.variance.detach().cpu().numpy()
        var_z = np.clip(var_z, 0.0, None)

        return {
            "exp_density_z_mean":   mean_z[:, 0],
            "exp_density_z_var":    var_z[:, 0],
            "exp_density_std_norm": np.sqrt(var_z[:, 0]),
            "diff_z_mean":          mean_z[:, 1],
            "diff_z_var":           var_z[:, 1],
            "diff_std_norm":        np.sqrt(var_z[:, 1]),
        }

    # ------------------------------------------------------------------
    # Checkpoint round-trip
    # ------------------------------------------------------------------

    def to_checkpoint(
        self,
        regime: str,
        label_scaler_scope: str,
        original_indices: list[int],
        model_name: str,
        iteration: int,
        extra: dict | None = None,
    ) -> dict:
        """Build a serializable checkpoint dict for this expert."""
        ckpt = {
            "model_state_dict":         self.model.state_dict(),
            "likelihood_state_dict":    self.likelihood.state_dict(),
            "feature_normalizer_stats": self.feature_normalizer_stats,
            "label_scaler1":            self.label_scaler1,
            "label_scaler2":            self.label_scaler2,
            "label_columns":            self.label_columns,
            "feature_columns":          self.feature_columns,
            "regime":                   regime,
            "transform":                self.transform,
            "label_scaler_scope":       label_scaler_scope,
            "original_indices":         list(original_indices),
            "model_name":               model_name,
            "iter":                     iteration,
        }
        if extra:
            ckpt.update(extra)
        return ckpt

    def save_checkpoint(
        self,
        path: str,
        regime: str,
        label_scaler_scope: str,
        original_indices: list[int],
        model_name: str,
        iteration: int,
        extra: dict | None = None,
    ) -> None:
        save_dir = os.path.dirname(path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(
            self.to_checkpoint(regime, label_scaler_scope, original_indices, model_name, iteration, extra),
            path,
        )

    @classmethod
    def from_checkpoint(
        cls,
        ckpt: dict,
        features_train_file: str,
        labels_train_file: str,
    ) -> "GPRExpert":
        """
        Reconstruct a live expert from a checkpoint + training data files.

        ExactGP needs its training tensors to compute predictive posteriors, so
        we rebuild them from the stored `original_indices` using the checkpoint's
        own normalizer / scalers / transform — guaranteeing the loaded expert
        sees the same training data the saved expert was fit on.
        """
        label_columns = list(ckpt["label_columns"])
        transform = ckpt["transform"]
        original_indices = ckpt["original_indices"]

        features_all = pd.read_csv(features_train_file)
        labels_all = pd.read_csv(labels_train_file)
        feature_columns = ckpt.get("feature_columns", features_all.columns.tolist())

        features_raw = features_all.iloc[original_indices].reset_index(drop=True)
        labels_raw = labels_all.iloc[original_indices].reset_index(drop=True)

        feat_conv = convert_features(features_raw[feature_columns])
        feat_norm = apply_feature_normalizer(feat_conv, ckpt["feature_normalizer_stats"])
        train_x = torch.tensor(feat_norm.to_numpy(), dtype=torch.float32)

        scaler1 = ckpt["label_scaler1"]
        scaler2 = ckpt["label_scaler2"]
        lab_prepared = _prepare_label_array(labels_raw, label_columns, transform)
        lab_scaled = lab_prepared.copy()
        lab_scaled[:, 0] = scaler1.transform(lab_prepared[:, [0]]).ravel()
        lab_scaled[:, 1] = scaler2.transform(lab_prepared[:, [1]]).ravel()
        train_y = torch.tensor(lab_scaled, dtype=torch.float32)

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
        model = MultitaskGPRegressionModel(train_x, train_y, likelihood, num_tasks=2)
        model.load_state_dict(ckpt["model_state_dict"])
        likelihood.load_state_dict(ckpt["likelihood_state_dict"])
        model.eval()
        likelihood.eval()

        return cls(
            model, likelihood, ckpt["feature_normalizer_stats"],
            scaler1, scaler2, transform, label_columns, feature_columns,
            label_scaler_scope=ckpt.get("label_scaler_scope"),
            model_name=ckpt.get("model_name"),
            iteration=ckpt.get("iter"),
            regime=ckpt.get("regime"),
        )
