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
import torch

from al_pipeline.surrogates.base import PoolPosterior, Surrogate


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

    @property
    def means(self) -> np.ndarray:
        return self._means

    @property
    def stds(self) -> np.ndarray:
        return self._stds

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

    Construction is split from prediction so the heavy `load_models` work can
    happen once per GA candidate (it already does — see
    `run_ga.run_one_candidate`).
    """

    def __init__(self, *, mode: str, model_bundle: dict[str, Any], obj1: str, obj2: str):
        if mode not in ("gpr_multitask", "gpr_singletask"):
            raise ValueError(f"Unsupported GPR mode: {mode!r}")
        self._mode = mode
        self._bundle = model_bundle
        self._obj1 = obj1
        self._obj2 = obj2

    @property
    def supports_joint_sampling(self) -> bool:
        return self._mode == "gpr_multitask"

    def predict_pool(self, Xn: np.ndarray) -> PoolPosterior:
        X_tensor = (
            Xn if torch.is_tensor(Xn) else torch.tensor(np.asarray(Xn).copy(), dtype=torch.float32)
        )

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


def make_surrogate(cfg, model_bundle: dict[str, Any]) -> Surrogate:
    """
    Dispatch to the right `Surrogate` based on `cfg.train_model_type`.

    The MoE surrogate will land here as a new `elif` branch on a future
    feat/moe-core branch.
    """
    t = cfg.train_model_type
    if t in ("gpr_multitask", "gpr_singletask"):
        return GlobalGPRSurrogate(
            mode=t, model_bundle=model_bundle, obj1=cfg.obj1, obj2=cfg.obj2
        )
    if t == "dnn":
        raise NotImplementedError("DNN surrogate not implemented yet.")
    raise ValueError(f"Unknown train_model_type={t}")
