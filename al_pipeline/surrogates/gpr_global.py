"""
Global-GPR concrete `Surrogate` — wraps the existing `gpr_multitask` and
`gpr_singletask` paths behind the new abstract interface.

Behavior is preserved bit-for-bit relative to the pre-refactor code in
`run_ga.py::_predict_mu_std` and `ehvi.monte_carlo_ehvi_batch`:

  - Multitask: one call to `model(Xn)` → cached `MultivariateNormal`. `.means`
    and `.stds` come from `.mean` / `.stddev`; `.sample(n)` calls
    `posterior.rsample(torch.Size([n]))`. Same `fast_pred_var()` context the
    GA used before.
  - Singletask: two separate `model_i(Xn)` calls, one per objective, in
    `(obj1, obj2)` column order. `.sample(...)` raises — single-task posteriors
    have no cross-objective covariance, so MC-EHVI would be a lie.

`make_surrogate(cfg, model_bundle)` is the dispatch from `cfg.train_model_type`
to the right concrete surrogate. New surrogate types (MoE, DNN) hook in by
extending this factory.
"""
from __future__ import annotations

from typing import Any

import gpytorch
import numpy as np
import pandas as pd
import torch

from al_pipeline.data_prep.data_loading import convert_and_normalize_features
from al_pipeline.surrogates.base import DesignPrediction, PoolPosterior, Surrogate


class _MultitaskPoolPosterior(PoolPosterior):
    """Wraps a cached gpytorch MultivariateNormal over (B, 2) joint outputs."""

    def __init__(self, posterior: gpytorch.distributions.MultitaskMultivariateNormal):
        # Cache means/stds eagerly — they're cheap to read off the posterior
        # and the analytic consumer will always want them. Keeps `.sample(n)`
        # the only operation that touches the (potentially expensive) joint
        # covariance.
        with torch.no_grad():
            self._means = posterior.mean.detach().cpu().numpy()
            self._stds = posterior.stddev.detach().cpu().numpy()
        self._posterior = posterior
        self._cov_cache: np.ndarray | None = None

    @property
    def means(self) -> np.ndarray:
        return self._means

    @property
    def stds(self) -> np.ndarray:
        return self._stds

    @property
    def covariance(self) -> np.ndarray:
        # Extract per-candidate (T, T) blocks from the full (B*T, B*T) joint
        # covariance. Mirrors the reshape used by augmentation.get_cand_stats
        # so any consumer that switches from the direct-GP path to the
        # surrogate gets identical numbers.
        if self._cov_cache is None:
            with torch.no_grad():
                B, T = self._means.shape
                cov = self._posterior.covariance_matrix   # (B*T, B*T)
                cov = cov.reshape(B, T, B, T)
                per_cand = cov[torch.arange(B), :, torch.arange(B), :]   # (B, T, T)
                self._cov_cache = per_cand.detach().cpu().numpy()
        return self._cov_cache

    def sample(self, n_samples: int) -> torch.Tensor:
        with torch.no_grad():
            return self._posterior.rsample(torch.Size([n_samples]))


class _SingletaskPoolPosterior(PoolPosterior):
    """Two independent per-objective posteriors stacked into (B, 2)."""

    def __init__(
        self,
        post1: gpytorch.distributions.MultivariateNormal,
        post2: gpytorch.distributions.MultivariateNormal,
    ):
        with torch.no_grad():
            m1 = post1.mean.detach().cpu().numpy().reshape(-1)
            m2 = post2.mean.detach().cpu().numpy().reshape(-1)
            s1 = post1.stddev.detach().cpu().numpy().reshape(-1)
            s2 = post2.stddev.detach().cpu().numpy().reshape(-1)
        self._means = np.column_stack([m1, m2])
        self._stds = np.column_stack([s1, s2])

    @property
    def means(self) -> np.ndarray:
        return self._means

    @property
    def stds(self) -> np.ndarray:
        return self._stds

    @property
    def covariance(self) -> np.ndarray:
        # No cross-objective structure available — return a diagonal fallback
        # so the pessimism math still runs (it degrades to the marginal-var
        # case, which is what the pre-refactor singletask path effectively
        # did in augmentation.py).
        B, T = self._means.shape
        cov = np.zeros((B, T, T), dtype=self._stds.dtype)
        for t in range(T):
            cov[:, t, t] = self._stds[:, t] ** 2
        return cov

    def sample(self, n_samples: int) -> torch.Tensor:
        # Independent per-objective sampling would not capture cross-objective
        # covariance, and MC-EHVI assumes it does. The pre-refactor code raised
        # the same error from `monte_carlo_ehvi_batch` when called with
        # singletask; keep that contract.
        raise NotImplementedError(
            "Single-task GPR has no cross-objective covariance; MC-EHVI requires "
            "a joint posterior. Use train_model_type='gpr_multitask' for MC-EHVI."
        )


class GlobalGPRSurrogate(Surrogate):
    """
    Surrogate backed by the existing global GPR (multitask or singletask).

    Stores the *global* feature-normalization stats — produced when the global
    GPR was trained — and applies them to raw features inside `predict_pool`.
    That keeps the GA side stupid: it hands the surrogate a DataFrame of raw
    features from the featurizer and gets back a posterior, with no knowledge
    of what preprocessing the surrogate happens to need.
    """

    def __init__(
        self,
        *,
        mode: str,
        model_bundle: dict[str, Any],
        normalization_stats: dict,
        obj1: str,
        obj2: str,
    ):
        if mode not in ("gpr_multitask", "gpr_singletask"):
            raise ValueError(f"Unsupported GPR mode: {mode!r}")
        self._mode = mode
        self._bundle = model_bundle
        self._stats = normalization_stats
        self._obj1 = obj1
        self._obj2 = obj2

    @property
    def supports_joint_sampling(self) -> bool:
        return self._mode == "gpr_multitask"

    @property
    def model_bundle(self) -> dict[str, Any]:
        """Live reference to the underlying model dict.

        Mutable — the retrospective's kriging-believer path re-conditions
        the GP by swapping in retrained `model`/`likelihood` objects here."""
        return self._bundle

    @property
    def normalization_stats(self) -> dict:
        """Global feature normalization stats used by `_normalize`."""
        return self._stats

    @property
    def mode(self) -> str:
        return self._mode

    def _normalize(self, X_raw: pd.DataFrame) -> torch.Tensor:
        """Convert + standardize via the stored global stats, return a torch tensor."""
        Xraw = X_raw.to_numpy(dtype=np.float32)
        Xn = convert_and_normalize_features(Xraw, train=False, stats=self._stats)
        return torch.tensor(np.asarray(Xn, dtype=np.float32), dtype=torch.float32)

    def predict_pool(self, X_raw: pd.DataFrame) -> PoolPosterior:
        X_tensor = self._normalize(X_raw)

        if self._mode == "gpr_multitask":
            model = self._bundle["model"]
            model.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                post = model(X_tensor)
            return _MultitaskPoolPosterior(post)

        # singletask
        models = self._bundle["models"]
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            post1 = models[self._obj1](X_tensor)
            post2 = models[self._obj2](X_tensor)
        return _SingletaskPoolPosterior(post1, post2)

    def predict_design(
        self, X_raw: pd.DataFrame, *, regime: str | None = None,
    ) -> DesignPrediction:
        """Beam-facing prediction for the global GPR baseline.

        Returns z-space mean + std; ``phys_mean`` is left as ``None`` because
        the global GPR training path currently writes the pre-scaled
        ``labels_norm_csv`` but does *not* persist the fitted
        `PowerTransformer` instances. Beam's ``--policy global`` cannot
        therefore recover physical values without refitting scalers from the
        raw labels — the same drift risk Row 8 is removing. A follow-up
        branch will add scaler persistence to `kfold_training`; until then,
        MoE surrogates are the only path with a fully-specified physical
        inversion (persisted in `GPRExpert` checkpoints).

        The ``regime`` kwarg is accepted for signature compatibility with the
        ABC but ignored — this surrogate has no per-regime experts to skip.
        """
        del regime  # single-expert surrogate; nothing to skip.
        pool = self.predict_pool(X_raw)
        z_mean = np.asarray(pool.means, dtype=np.float64)
        z_std = np.asarray(pool.stds, dtype=np.float64)
        return DesignPrediction(
            z_mean=z_mean,
            z_std=z_std,
            sigma_z=z_std,
            phys_mean=None,
            phys_std=None,
            p_ps=None,
            per_expert=None,
        )


def make_surrogate(
    cfg,
    *,
    model_bundle: dict[str, Any] | None = None,
    normalization_stats: dict | None = None,
    moe_bundle=None,
    moe_policy: str = "soft",
    moe_threshold: float = 0.5,
) -> Surrogate:
    """
    Dispatch to the right `Surrogate` based on `cfg.train_model_type`.

    Parameters
    ----------
    cfg : ALConfig
        Used for train_model_type, obj1, obj2 dispatch.
    model_bundle : dict | None
        Loaded GPR bundle. Required for `train_model_type` in {gpr_multitask,
        gpr_singletask}; ignored for moe.
    normalization_stats : dict | None
        Global feature normalization stats. Required for the GPR modes;
        ignored for moe (each MoE expert has its own normalizer).
    moe_bundle : MoEBundle | None
        Required when `train_model_type == 'moe'`.
    moe_policy, moe_threshold
        MoE gate-blend configuration. Only used when `train_model_type == 'moe'`.
    """
    t = cfg.train_model_type
    if t in ("gpr_multitask", "gpr_singletask"):
        if model_bundle is None or normalization_stats is None:
            raise ValueError(f"make_surrogate({t!r}) requires model_bundle + normalization_stats")
        return GlobalGPRSurrogate(
            mode=t,
            model_bundle=model_bundle,
            normalization_stats=normalization_stats,
            obj1=cfg.obj1,
            obj2=cfg.obj2,
        )
    if t == "moe":
        if moe_bundle is None:
            raise ValueError("make_surrogate('moe') requires moe_bundle")
        # Imported here to avoid a base-module -> moe import cycle; moe imports
        # Surrogate / PoolPosterior from base.
        from al_pipeline.surrogates.moe import MoESurrogate
        return MoESurrogate(
            bundle=moe_bundle,
            policy=moe_policy,
            threshold=moe_threshold,
        )
    if t == "dnn":
        raise NotImplementedError("DNN surrogate not implemented yet.")
    raise ValueError(f"Unknown train_model_type={t}")
