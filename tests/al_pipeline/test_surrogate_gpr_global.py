"""
Tests for `GlobalGPRSurrogate` — the concrete `Surrogate` that wraps the existing
multitask / singletask GPR consumed by the acquisition loop.

The point of the surrogate ABC is that `GA` calls `surrogate.predict_pool(X)` instead
of poking the GP model directly. So the round-trip we care about is:

  raw GP output  ==  surrogate.predict_pool(X).means / .stds / .sample(n)

for both multitask and singletask GPR. If those agree, the surrogate refactor is
behavior-preserving by construction — `run_ga.py` consumes the same numbers it did
before the abstraction landed.
"""
from __future__ import annotations

import types

import gpytorch
import numpy as np
import pytest
import torch

from al_pipeline.surrogates import GlobalGPRSurrogate, make_surrogate
from al_pipeline.surrogates.base import PoolPosterior, Surrogate
from al_pipeline.training.ml_models import (
    GPRegressionModel,
    MultitaskGPRegressionModel,
)


# ---------- small fixtures ----------

def _toy_multitask_bundle(n_train: int = 16, n_feat: int = 5, seed: int = 0):
    """Build a tiny untrained multitask GPR + matched bundle dict."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    X = torch.tensor(rng.standard_normal((n_train, n_feat)), dtype=torch.float32)
    y = torch.tensor(rng.standard_normal((n_train, 2)), dtype=torch.float32)
    lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    model = MultitaskGPRegressionModel(X, y, lik, num_tasks=2)
    model.eval()
    lik.eval()
    return {"model": model, "likelihood": lik, "X_train": X, "y_train": y}


def _toy_singletask_bundle(n_train: int = 16, n_feat: int = 5, seed: int = 0):
    """Build two tiny untrained singletask GPRs (one per objective)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    X = torch.tensor(rng.standard_normal((n_train, n_feat)), dtype=torch.float32)
    y_obj1 = torch.tensor(rng.standard_normal(n_train), dtype=torch.float32)
    y_obj2 = torch.tensor(rng.standard_normal(n_train), dtype=torch.float32)

    models, likelihoods, y_trains = {}, {}, {}
    for label, y in (("obj1", y_obj1), ("obj2", y_obj2)):
        lik = gpytorch.likelihoods.GaussianLikelihood()
        m = GPRegressionModel(X, y, lik)
        m.eval()
        lik.eval()
        models[label] = m
        likelihoods[label] = lik
        y_trains[label] = y
    return {"models": models, "likelihoods": likelihoods, "X_train": X, "y_train": y_trains}


def _fake_cfg(model_type: str, obj1: str = "obj1", obj2: str = "obj2"):
    """Cheap stand-in for ALConfig with just the attributes make_surrogate reads."""
    return types.SimpleNamespace(train_model_type=model_type, obj1=obj1, obj2=obj2)


# ---------- multitask: surrogate matches raw GP ----------

def test_multitask_predict_pool_matches_raw_model():
    """The surrogate's means/stds must equal what `model(X).mean / .stddev` returns
    directly. If not, anything downstream that relied on the raw output silently
    gets different numbers after the refactor."""
    bundle = _toy_multitask_bundle()
    Xn = np.random.RandomState(1).randn(7, 5).astype(np.float32)

    cfg = _fake_cfg("gpr_multitask")
    sur = make_surrogate(cfg, bundle)
    pool = sur.predict_pool(Xn)

    # Direct path: what run_ga used to do before the surrogate
    model = bundle["model"]
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        post = model(torch.as_tensor(Xn, dtype=torch.float32))
        mu_raw = post.mean.cpu().numpy()
        sd_raw = post.stddev.cpu().numpy()

    np.testing.assert_allclose(pool.means, mu_raw, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(pool.stds, sd_raw, rtol=1e-6, atol=1e-6)


def test_multitask_sample_shape_and_dtype():
    """sample(n) must produce (n, B, 2) torch float32 — the contract the MC EHVI
    consumer assumes (it casts to fp64 itself)."""
    bundle = _toy_multitask_bundle()
    Xn = np.random.RandomState(2).randn(4, 5).astype(np.float32)
    sur = make_surrogate(_fake_cfg("gpr_multitask"), bundle)

    pool = sur.predict_pool(Xn)
    n = 8
    samples = pool.sample(n)
    assert samples.shape == (n, 4, 2)
    assert samples.dtype == torch.float32


def test_multitask_sample_reproducibility_with_seed():
    """Two `sample(n)` calls under the same torch seed must produce identical draws —
    pins down that the surrogate doesn't introduce hidden RNG state."""
    bundle = _toy_multitask_bundle()
    Xn = np.random.RandomState(3).randn(3, 5).astype(np.float32)
    sur = make_surrogate(_fake_cfg("gpr_multitask"), bundle)
    pool = sur.predict_pool(Xn)

    torch.manual_seed(99)
    a = pool.sample(4).clone()
    torch.manual_seed(99)
    b = pool.sample(4).clone()
    assert torch.equal(a, b)


def test_multitask_supports_joint_sampling_flag():
    sur = make_surrogate(_fake_cfg("gpr_multitask"), _toy_multitask_bundle())
    assert sur.supports_joint_sampling is True


# ---------- singletask: surrogate stacks per-objective posteriors correctly ----------

def test_singletask_predict_pool_matches_per_objective_posteriors():
    """Columns of `pool.means` / `pool.stds` must match the per-objective posteriors,
    in `(cfg.obj1, cfg.obj2)` order — independent of dict iteration order."""
    bundle = _toy_singletask_bundle()
    Xn = np.random.RandomState(4).randn(5, 5).astype(np.float32)
    cfg = _fake_cfg("gpr_singletask", obj1="obj1", obj2="obj2")

    sur = make_surrogate(cfg, bundle)
    pool = sur.predict_pool(Xn)

    Xt = torch.as_tensor(Xn, dtype=torch.float32)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        p1 = bundle["models"]["obj1"](Xt)
        p2 = bundle["models"]["obj2"](Xt)

    np.testing.assert_allclose(pool.means[:, 0], p1.mean.cpu().numpy().ravel(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(pool.means[:, 1], p2.mean.cpu().numpy().ravel(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(pool.stds[:, 0], p1.stddev.cpu().numpy().ravel(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(pool.stds[:, 1], p2.stddev.cpu().numpy().ravel(), rtol=1e-6, atol=1e-6)


def test_singletask_sample_raises_not_implemented():
    """Single-task GPR has no cross-objective covariance — joint sampling has to
    error out. This is the same behavior as the pre-refactor code, which raised
    inside `monte_carlo_ehvi_batch`."""
    sur = make_surrogate(_fake_cfg("gpr_singletask"), _toy_singletask_bundle())
    pool = sur.predict_pool(np.random.RandomState(5).randn(2, 5).astype(np.float32))
    with pytest.raises(NotImplementedError):
        pool.sample(4)


def test_singletask_supports_joint_sampling_flag():
    sur = make_surrogate(_fake_cfg("gpr_singletask"), _toy_singletask_bundle())
    assert sur.supports_joint_sampling is False


# ---------- factory + type hygiene ----------

def test_make_surrogate_returns_global_gpr_for_known_modes():
    """The factory dispatches on `cfg.train_model_type` — both GPR modes must
    yield a `GlobalGPRSurrogate`. (MoE will land as a new dispatch branch.)"""
    sur_mt = make_surrogate(_fake_cfg("gpr_multitask"), _toy_multitask_bundle())
    sur_st = make_surrogate(_fake_cfg("gpr_singletask"), _toy_singletask_bundle())
    assert isinstance(sur_mt, GlobalGPRSurrogate)
    assert isinstance(sur_st, GlobalGPRSurrogate)
    assert isinstance(sur_mt, Surrogate)


def test_make_surrogate_rejects_unknown_mode():
    with pytest.raises(ValueError):
        make_surrogate(_fake_cfg("not-a-real-mode"), _toy_multitask_bundle())


def test_global_gpr_surrogate_rejects_bad_mode_kwarg():
    """Direct ctor path also validates the mode string — defense in depth."""
    with pytest.raises(ValueError):
        GlobalGPRSurrogate(mode="bogus", model_bundle={}, obj1="o1", obj2="o2")


def test_predict_pool_returns_pool_posterior_instance():
    sur = make_surrogate(_fake_cfg("gpr_multitask"), _toy_multitask_bundle())
    pool = sur.predict_pool(np.zeros((2, 5), dtype=np.float32))
    assert isinstance(pool, PoolPosterior)
