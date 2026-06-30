"""
Mixture-of-experts surrogate.

The pieces:

  - `build_rf_features` / `classifier_p_ps` — turn raw features into the RF's
    expected design matrix (converted but NOT standardized; RF is scale-
    invariant and benefits from semantically meaningful AA fractions) and
    read out P(PS | x) with single-class-fold guards.

  - `MoEBundle` — load + validate RF + PS + nonPS GPR experts. There is
    deliberately NO "all" expert in the bundle: the AL acquisition only
    blends PS + nonPS via the gate, and diagnostic consumers that want a
    global comparison can stand up a `GlobalGPRSurrogate` separately (it
    already exists per iter). Avoids training the same model twice.

  - `MoEPoolPosterior` — a `PoolPosterior` over (B, 2) candidate predictions
    that blends PS / nonPS experts by the RF gate. Two policies:
      * "soft"  : means via combine_soft, vars via soft_mixture_variance,
                  samples via per-draw Bernoulli mixture sampling.
      * "hard"  : deterministic gate per candidate; means/vars/samples take
                  the chosen expert's joint posterior wholesale.

  - `MoESurrogate` — the `Surrogate` ABC implementation. Construction asserts
    `label_scaler_scope == 'all'` because mixing in z-space is only meaningful
    when every expert uses the same label scaler. (Per-regime scaling is fine
    for beam search; it gets a different consumer.)

The AL loop's existing analytic and MC EHVI consumers work unchanged —
`MoESurrogate.predict_pool` returns a `PoolPosterior` exactly like
`GlobalGPRSurrogate` does.
"""
from __future__ import annotations

import os
import pickle
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch

from al_pipeline.data_prep.data_loading import convert_features
from al_pipeline.surrogates.base import PoolPosterior, Surrogate
from al_pipeline.surrogates.gpr_expert import GPRExpert
from al_pipeline.surrogates.moe_combine import soft_mixture_variance

EXPERTS = ("ps", "nonps")
Policy = Literal["soft", "hard"]


# ---------------------------------------------------------------------------
# RF feature pipeline
# ---------------------------------------------------------------------------

def build_rf_features(
    features_df: pd.DataFrame,
    rf_raw_feature_columns: list[str],
    rf_converted_feature_columns: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Build the RF design matrix from raw features.

    Length-normalized "converted" features (AA counts -> fractions) but NOT
    standardized. RF is scale-invariant so standardization would only add
    coupling between RF and the GPR feature normalizers without buying
    anything. Raw and converted column lists are tracked explicitly so a
    future change to `convert_features` cannot silently shift columns.
    """
    missing_raw = [c for c in rf_raw_feature_columns if c not in features_df.columns]
    if missing_raw:
        raise ValueError(f"Features missing raw columns required by RF: {missing_raw}")

    features_raw = features_df[list(rf_raw_feature_columns)]
    features_conv = convert_features(features_raw)

    if rf_converted_feature_columns is None:
        rf_converted_feature_columns = features_conv.columns.tolist()
    else:
        missing_conv = [c for c in rf_converted_feature_columns if c not in features_conv.columns]
        if missing_conv:
            raise ValueError(f"Converted features missing columns required by RF: {missing_conv}")

    X = features_conv[list(rf_converted_feature_columns)].to_numpy()
    return X, list(rf_converted_feature_columns)


def classifier_p_ps(rf, X: np.ndarray) -> np.ndarray:
    """
    P(PS | x) from an sklearn RF, robust to a single-class training fold.

    If the training data for the RF only saw nonPS rows (label 0), `predict_proba`
    has no PS column and we return zeros. Better than crashing on degenerate
    early-iteration data.
    """
    classes = list(rf.classes_)
    proba = rf.predict_proba(X)
    if 1 in classes:
        return proba[:, classes.index(1)]
    return np.zeros(X.shape[0], dtype=np.float64)


# ---------------------------------------------------------------------------
# RF bundle (de)serialization
# ---------------------------------------------------------------------------

def save_rf_bundle(
    path: str,
    rf,
    *,
    rf_raw_feature_columns: list[str],
    rf_converted_feature_columns: list[str],
    ps_definition: str,
    random_state: int,
    threshold: float,
    model_name: str,
    iteration: int,
    transform: str,
    label_scaler_scope: str,
    best_params: dict | None = None,
    extra: dict | None = None,
) -> None:
    """
    Pickle the RF classifier with all metadata the loader and validator need.

    The bundle is the source of truth for the gate's feature contract — every
    consumer (`MoESurrogate`, beam-search diagnostic) reads the raw and
    converted column lists from here, not from a separate config.
    """
    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    bundle = {
        "classifier":                   rf,
        "rf_raw_feature_columns":       list(rf_raw_feature_columns),
        "rf_converted_feature_columns": list(rf_converted_feature_columns),
        "rf_feature_space":             "converted_unstandardized",
        "ps_definition":                ps_definition,
        "random_state":                 random_state,
        "threshold":                    threshold,
        "model_name":                   model_name,
        "iter":                         iteration,
        "transform":                    transform,
        "label_scaler_scope":           label_scaler_scope,
    }
    if best_params is not None:
        bundle["best_params"] = best_params
    if extra:
        bundle.update(extra)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def load_rf_bundle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# MoE bundle
# ---------------------------------------------------------------------------

class MoEBundle:
    """
    A loaded MoE: RF classifier + PS + nonPS GPR experts + agreed metadata.

    No "all" expert — the AL acquisition path only blends PS + nonPS via the
    gate. Diagnostic / beam-search consumers that want a global comparison
    stand up a `GlobalGPRSurrogate` alongside the MoE one (it already exists
    per iter).

    Construct via `from_components(...)` (in-process: tests, training pipeline)
    or `from_checkpoints(...)` (future, lands with feat/moe-training).
    """

    def __init__(
        self,
        rf_bundle: dict,
        ps_expert: GPRExpert,
        nonps_expert: GPRExpert,
        label_scaler_scope: str,
        transform: str,
    ):
        self.rf_bundle = rf_bundle
        self.rf = rf_bundle["classifier"]
        # Fall back to legacy bundle key 'feature_columns' for older checkpoints
        # (MCSC pre-rf_raw_feature_columns rename); harmless for new bundles.
        self.rf_raw_feature_columns = rf_bundle.get(
            "rf_raw_feature_columns", rf_bundle.get("feature_columns"),
        )
        self.rf_converted_feature_columns = rf_bundle.get("rf_converted_feature_columns")
        self.ps_expert = ps_expert
        self.nonps_expert = nonps_expert
        self.label_scaler_scope = label_scaler_scope
        self.transform = transform

    @staticmethod
    def _validate_metadata(rf_bundle: dict, experts: dict[str, GPRExpert]) -> None:
        """
        Check every expert's transform / scope / model_name / iter against the RF.

        Provenance is stamped into checkpoints and the bundle refuses to load
        if anything disagrees. Catches silent mismatches at load time rather
        than at prediction time.
        """
        rf_transform = rf_bundle.get("transform")
        rf_scope = rf_bundle.get("label_scaler_scope")
        rf_model = rf_bundle.get("model_name")
        rf_iter = rf_bundle.get("iter")

        errors = []
        for name, ex in experts.items():
            if ex.regime is not None and ex.regime != name:
                errors.append(f"{name}-slot checkpoint has regime={ex.regime!r} (expected {name!r})")
            if ex.transform != rf_transform:
                errors.append(f"{name}-expert transform={ex.transform!r} != RF transform={rf_transform!r}")
            if ex.label_scaler_scope != rf_scope:
                errors.append(
                    f"{name}-expert label_scaler_scope={ex.label_scaler_scope!r} "
                    f"!= RF label_scaler_scope={rf_scope!r}"
                )
            if ex.model_name is not None and rf_model is not None and ex.model_name != rf_model:
                errors.append(f"{name}-expert model_name={ex.model_name!r} != RF model_name={rf_model!r}")
            if ex.iteration is not None and rf_iter is not None and ex.iteration != rf_iter:
                errors.append(f"{name}-expert iter={ex.iteration!r} != RF iter={rf_iter!r}")

        if errors:
            raise ValueError("MoE metadata mismatch:\n  " + "\n  ".join(errors))

    @classmethod
    def from_components(
        cls,
        rf_bundle: dict,
        ps_expert: GPRExpert,
        nonps_expert: GPRExpert,
    ) -> "MoEBundle":
        """Build a bundle from already-instantiated parts."""
        experts = {"ps": ps_expert, "nonps": nonps_expert}
        cls._validate_metadata(rf_bundle, experts)
        scope = rf_bundle.get("label_scaler_scope", "regime")
        transform = rf_bundle.get("transform", ps_expert.transform)
        return cls(rf_bundle, ps_expert, nonps_expert, scope, transform)

    @classmethod
    def from_checkpoints(
        cls,
        rf_pkl: str,
        ps_gpr_ckpt: str,
        nonps_gpr_ckpt: str,
        features_train_file: str,
        labels_train_file: str,
        *,
        expected_transform: str | None = None,
        expected_label_scaler_scope: str | None = None,
        expected_model_name: str | None = None,
        expected_iter: int | None = None,
    ) -> "MoEBundle":
        """
        Load an RF bundle + PS + nonPS expert checkpoints into a live MoE.

        The expert ExactGPs need their training tensors to compute predictive
        posteriors, so we reload the raw features + labels CSVs and let each
        `GPRExpert.from_checkpoint` rebuild its own train tensors from the
        stored `original_indices`.

        Optional `expected_*` kwargs let callers (e.g. the AL CLI) assert that
        the loaded bundle matches the expected iter / transform / scope,
        catching stale checkpoints early.
        """
        for pth in (rf_pkl, ps_gpr_ckpt, nonps_gpr_ckpt, features_train_file, labels_train_file):
            if not os.path.exists(pth):
                raise FileNotFoundError(f"Required MoE artifact not found: {pth}")

        rf_bundle = load_rf_bundle(rf_pkl)
        # `weights_only=False` lets us pickle-load the sklearn label scalers
        # that travel with each expert checkpoint. Same affordance MCSC needed.
        ps_ckpt = torch.load(ps_gpr_ckpt, map_location="cpu", weights_only=False)
        nps_ckpt = torch.load(nonps_gpr_ckpt, map_location="cpu", weights_only=False)
        ps_expert = GPRExpert.from_checkpoint(ps_ckpt, features_train_file, labels_train_file)
        nps_expert = GPRExpert.from_checkpoint(nps_ckpt, features_train_file, labels_train_file)

        # Run the standard validator over scope / transform / iter / model_name.
        bundle = cls.from_components(rf_bundle, ps_expert, nps_expert)

        # Optional caller assertions on top of the structural check.
        rf_transform = rf_bundle.get("transform")
        rf_scope = rf_bundle.get("label_scaler_scope")
        rf_model = rf_bundle.get("model_name")
        rf_iter = rf_bundle.get("iter")
        mismatches = []
        if expected_transform is not None and rf_transform != expected_transform:
            mismatches.append(f"transform={rf_transform!r} != expected {expected_transform!r}")
        if expected_label_scaler_scope is not None and rf_scope != expected_label_scaler_scope:
            mismatches.append(f"scope={rf_scope!r} != expected {expected_label_scaler_scope!r}")
        if expected_model_name is not None and rf_model != expected_model_name:
            mismatches.append(f"model_name={rf_model!r} != expected {expected_model_name!r}")
        if expected_iter is not None and rf_iter != expected_iter:
            mismatches.append(f"iter={rf_iter!r} != expected {expected_iter!r}")
        if mismatches:
            raise ValueError("MoE checkpoint mismatch:\n  " + "\n  ".join(mismatches))
        return bundle


# ---------------------------------------------------------------------------
# MoE pool posterior + surrogate
# ---------------------------------------------------------------------------

class MoEPoolPosterior(PoolPosterior):
    """
    PoolPosterior over (B, 2) that blends PS / nonPS experts by the RF gate.

    Caches both expert posteriors at construction time. `.sample(n)` draws joint
    samples from each expert's MultitaskMultivariateNormal once per call and
    routes them through the policy:

      * soft: per-draw Bernoulli(p_ps[j]) per candidate; sample comes from PS
        with probability p_ps[j], nonPS otherwise. Preserves bimodality.
      * hard: deterministic where p_ps >= threshold; one expert per candidate
        for all draws. No randomness in the gate.

    Per-candidate analytic summaries (`means`, `stds`) use:
      * soft: combine_soft / soft_mixture_variance per objective.
      * hard: deterministic select of the chosen expert's mean/var.

    Variances are clipped to >= 0 to guard tiny negative roundoff before
    sqrt — the analytic EHVI stripe math will produce NaNs otherwise.
    """

    def __init__(
        self,
        p_ps: np.ndarray,
        post_ps,
        post_nonps,
        policy: Policy = "soft",
        threshold: float = 0.5,
    ):
        if policy not in ("soft", "hard"):
            raise ValueError(f"Unknown policy={policy!r}; expected 'soft' or 'hard'")
        self._policy = policy
        self._threshold = float(threshold)
        self._post_ps = post_ps
        self._post_nonps = post_nonps

        # P(PS | x) per candidate, plus a torch view for sample routing.
        self._p_ps = np.asarray(p_ps, dtype=np.float64).reshape(-1)
        self._p_ps_torch = torch.tensor(self._p_ps, dtype=torch.float32)

        # Cache per-expert means/vars eagerly — both analytic policies need them
        # and they're cheap to read off the gpytorch posterior.
        with torch.no_grad():
            self._mu_ps = post_ps.mean.detach().cpu().numpy()
            self._var_ps = np.clip(post_ps.variance.detach().cpu().numpy(), 0.0, None)
            self._mu_nonps = post_nonps.mean.detach().cpu().numpy()
            self._var_nonps = np.clip(post_nonps.variance.detach().cpu().numpy(), 0.0, None)

        # Precompute blended means/vars per policy.
        if policy == "soft":
            p = self._p_ps[:, None]   # (B, 1) broadcasts over the 2 objectives
            self._means = p * self._mu_ps + (1.0 - p) * self._mu_nonps
            self._vars = soft_mixture_variance(
                p, self._mu_ps, self._var_ps, self._mu_nonps, self._var_nonps,
            )
            self._vars = np.clip(self._vars, 0.0, None)
        else:  # hard
            use_ps = (self._p_ps >= self._threshold)[:, None]   # (B, 1)
            self._means = np.where(use_ps, self._mu_ps, self._mu_nonps)
            self._vars = np.where(use_ps, self._var_ps, self._var_nonps)
            self._vars = np.clip(self._vars, 0.0, None)

        self._stds = np.sqrt(self._vars)

    @property
    def means(self) -> np.ndarray:
        return self._means

    @property
    def stds(self) -> np.ndarray:
        return self._stds

    def sample(self, n_samples: int) -> torch.Tensor:
        with torch.no_grad():
            s_ps = self._post_ps.rsample(torch.Size([n_samples]))      # (n, B, 2)
            s_nonps = self._post_nonps.rsample(torch.Size([n_samples]))

        if self._policy == "soft":
            # Per-draw, per-candidate Bernoulli mixture sample. broadcasts over
            # the 2-objective dim — same component pick applies to (obj1, obj2)
            # within a draw, which preserves the joint covariance from that
            # expert's posterior.
            p = self._p_ps_torch.unsqueeze(0).expand(n_samples, -1)   # (n, B)
            z = torch.bernoulli(p).bool().unsqueeze(-1)               # (n, B, 1)
        else:  # hard
            use_ps = (self._p_ps_torch >= self._threshold).unsqueeze(-1)   # (B, 1)
            z = use_ps.unsqueeze(0).expand(n_samples, -1, -1)              # (n, B, 1)
        return torch.where(z, s_ps, s_nonps)


class MoESurrogate(Surrogate):
    """
    `Surrogate` ABC implementation backed by an `MoEBundle`.

    Construction enforces `label_scaler_scope='all'` — mixing predictions in
    z-space requires every expert to use the same label scaler. Beam search
    has different needs and gets its own (future) consumer.

    The "all" (global) expert in the bundle is loaded but not used by this
    surrogate — only PS and nonPS experts blend by the gate. The global expert
    is kept for diagnostic / beam-search consumers; the bundle is the right
    place to hold it.
    """

    def __init__(
        self,
        bundle: MoEBundle,
        *,
        policy: Policy = "soft",
        threshold: float = 0.5,
    ):
        if bundle.label_scaler_scope != "all":
            raise ValueError(
                "MoESurrogate requires label_scaler_scope='all'; got "
                f"{bundle.label_scaler_scope!r}. Per-regime scaling is fine for "
                "beam search but not the AL acquisition path — z-space mixing "
                "is only meaningful with a shared label scaler."
            )
        if policy not in ("soft", "hard"):
            raise ValueError(f"Unknown policy={policy!r}; expected 'soft' or 'hard'")
        self._bundle = bundle
        self._policy = policy
        self._threshold = float(threshold)

    @property
    def supports_joint_sampling(self) -> bool:
        # Each expert's multitask GP supports joint sampling; the mixture
        # over them does too via per-draw component routing.
        return True

    @property
    def bundle(self) -> MoEBundle:
        """Underlying bundle — useful for diagnostic consumers that want the
        per-expert outputs alongside the blended ones."""
        return self._bundle

    def predict_pool(self, X_raw: pd.DataFrame) -> PoolPosterior:
        # RF gate: build the (converted-but-unstandardized) RF design matrix
        # using the bundle's stored column lists, then read out P(PS | x).
        X_rf, _ = build_rf_features(
            X_raw,
            self._bundle.rf_raw_feature_columns,
            self._bundle.rf_converted_feature_columns,
        )
        p_ps = classifier_p_ps(self._bundle.rf, X_rf)

        # Per-expert joint posteriors over (B, 2). Each expert applies its own
        # feature normalizer to X_raw inside `posterior(...)`. Computed once
        # per batch; MoEPoolPosterior reuses them across analytic + MC consumers.
        post_ps = self._bundle.ps_expert.posterior(X_raw)
        post_nonps = self._bundle.nonps_expert.posterior(X_raw)

        return MoEPoolPosterior(
            p_ps=p_ps,
            post_ps=post_ps,
            post_nonps=post_nonps,
            policy=self._policy,
            threshold=self._threshold,
        )
