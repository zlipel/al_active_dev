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
        # Place the inference tensor on the same device as the model's
        # training data so gpytorch's forward runs on GPU when the model
        # was loaded with device="cuda". The 460 KB round-trip per 4096-row
        # chunk is negligible (microseconds) vs. the GP kernel matmul.
        device = self.model.train_inputs[0].device
        return torch.tensor(feat_norm.to_numpy(), dtype=torch.float32, device=device)

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

    def inverse_scale_z(self, z: np.ndarray) -> np.ndarray:
        """Invert ``(B, 2)`` z-space values back to physical units.

        Uses the persisted `label_scaler1` / `label_scaler2` for each
        objective, then reverses the ``log`` pre-transform on the diff
        channel when applicable. Matches the semantics of the training-time
        `_prepare_label_array` used to fit the scalers, so a round trip
        (physical → z via train-time scaler → physical via `inverse_scale_z`)
        recovers the original physical value up to float rounding.

        Accepts either a ``(B, 2)`` array (both objectives) or a ``(B, 2)``
        sample from a joint z-posterior — the shape is preserved.
        """
        z = np.asarray(z, dtype=np.float64)
        if z.ndim != 2 or z.shape[1] != 2:
            raise ValueError(f"inverse_scale_z expects shape (B, 2); got {z.shape}")
        phys = np.empty_like(z)
        phys[:, 0] = self.label_scaler1.inverse_transform(z[:, [0]]).ravel()
        phys[:, 1] = self.label_scaler2.inverse_transform(z[:, [1]]).ravel()
        if self.transform == "log":
            # Undo the log(y + 1e-8) applied to the diff channel in
            # `_prepare_label_array` at training time.
            phys[:, 1] = np.exp(phys[:, 1]) - 1e-8
        return phys

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
        *,
        train_x_direct: "torch.Tensor | None" = None,
        train_y_direct: "torch.Tensor | None" = None,
    ) -> dict:
        """
        Build a serializable checkpoint dict for this expert.

        Base (per-iter) checkpoints supply `original_indices` and let the loader
        rebuild ExactGP train tensors from features_csv + labels_csv. Temp
        checkpoints (written during kriging-believer augmentation) supply
        `train_x_direct` + `train_y_direct` instead, because synthesized
        children don't have "raw" labels to reindex from a CSV. The loader
        prefers the direct tensors when present.
        """
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
        if train_x_direct is not None:
            ckpt["train_x_direct"] = train_x_direct.detach().cpu()
        if train_y_direct is not None:
            ckpt["train_y_direct"] = train_y_direct.detach().cpu()
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
        *,
        train_x_direct: "torch.Tensor | None" = None,
        train_y_direct: "torch.Tensor | None" = None,
    ) -> None:
        save_dir = os.path.dirname(path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(
            self.to_checkpoint(
                regime, label_scaler_scope, original_indices, model_name, iteration, extra,
                train_x_direct=train_x_direct, train_y_direct=train_y_direct,
            ),
            path,
        )

    @classmethod
    def from_checkpoint(
        cls,
        ckpt: dict,
        features_train_file: str,
        labels_train_file: str,
        *,
        device: "str | torch.device" = "cpu",
    ) -> "GPRExpert":
        """
        Reconstruct a live expert from a checkpoint + training data files.

        Two loader paths:
          1. If the checkpoint has `train_x_direct` + `train_y_direct` (temp
             checkpoints written during augmentation), use those directly.
             Synthesized kriging-believer children live in these tensors —
             they don't correspond to rows in the raw CSVs.
          2. Otherwise, use `original_indices` to slice features_csv +
             labels_csv, re-apply this expert's own normalizer / scalers /
             transform. Guarantees the loaded expert sees the same training
             data the saved expert was fit on.

        ``device`` places the reconstructed model + train tensors on the
        requested torch device (``"cpu"`` default; ``"cuda"`` / ``"cuda:0"``
        on GPU nodes). The checkpoint itself is deserialized on CPU first
        (via ``torch.load(..., map_location=device)`` in the caller) and
        then the tensors reconstructed here inherit the requested device.
        """
        torch_device = torch.device(device)

        label_columns = list(ckpt["label_columns"])
        transform = ckpt["transform"]
        scaler1 = ckpt["label_scaler1"]
        scaler2 = ckpt["label_scaler2"]

        features_all = pd.read_csv(features_train_file)
        feature_columns = ckpt.get("feature_columns", features_all.columns.tolist())

        if "train_x_direct" in ckpt and "train_y_direct" in ckpt:
            # Temp checkpoint (augmented): direct tensors are the source of truth.
            train_x = ckpt["train_x_direct"].float().to(torch_device)
            train_y = ckpt["train_y_direct"].float().to(torch_device)
        else:
            # Base checkpoint: reconstruct from CSV + original_indices.
            labels_all = pd.read_csv(labels_train_file)
            original_indices = ckpt["original_indices"]
            features_raw = features_all.iloc[original_indices].reset_index(drop=True)
            labels_raw = labels_all.iloc[original_indices].reset_index(drop=True)

            feat_conv = convert_features(features_raw[feature_columns])
            feat_norm = apply_feature_normalizer(feat_conv, ckpt["feature_normalizer_stats"])
            train_x = torch.tensor(
                feat_norm.to_numpy(), dtype=torch.float32, device=torch_device,
            )

            lab_prepared = _prepare_label_array(labels_raw, label_columns, transform)
            lab_scaled = lab_prepared.copy()
            lab_scaled[:, 0] = scaler1.transform(lab_prepared[:, [0]]).ravel()
            lab_scaled[:, 1] = scaler2.transform(lab_prepared[:, [1]]).ravel()
            train_y = torch.tensor(lab_scaled, dtype=torch.float32, device=torch_device)

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2).to(torch_device)
        model = MultitaskGPRegressionModel(train_x, train_y, likelihood, num_tasks=2).to(torch_device)
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
