"""
Tests for `GlobalGPRSurrogate` — the concrete `Surrogate` that wraps the existing
multitask / singletask GPR consumed by the acquisition loop.

The surrogate's new contract (since feat/moe-core) takes RAW features as a
DataFrame and normalizes internally using stored stats. The round-trip we
care about is:

  raw -> manual normalize -> raw GP output
  ==
  raw -> surrogate.predict_pool -> .means / .stds / .sample(n)

If those agree, the surrogate refactor is behavior-preserving by construction —
`run_ga.py` consumes the same numbers it did before the abstraction landed.
"""
from __future__ import annotations

import types

import gpytorch
import numpy as np
import pandas as pd
import pytest
import torch

from al_pipeline.data_prep.data_loading import convert_and_normalize_features
from al_pipeline.surrogates import GlobalGPRSurrogate, make_surrogate
from al_pipeline.surrogates.base import PoolPosterior, Surrogate
from al_pipeline.training.ml_models import (
    GPRegressionModel,
    MultitaskGPRegressionModel,
)


AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
FEATURE_COLUMNS = AMINO_ACIDS + [
    "length", "SCD", "SHD", "|net charge|", "sum lambda",
    "beads(+)", "beads(-)", "shan ent", "mol wt",
]


def _make_raw_features_df(n: int, seed: int) -> pd.DataFrame:
    """Realistic 29-column raw feature DataFrame (matches SequenceFeaturizer output)."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        L = int(rng.integers(20, 160 + 1))
        probs = rng.dirichlet(np.ones(20))
        counts = rng.multinomial(L, probs).astype(float)
        scd = float(rng.normal(0.0, 0.5))
        shd = float(rng.uniform(0.0, 2.5))
        net = float(rng.integers(0, 10))
        sum_lambda = float(rng.uniform(0.0, 3.5))
        beads_pos = float(rng.integers(0, max(1, L // 5)))
        beads_neg = float(rng.integers(0, max(1, L // 5)))
        shan_ent = float(rng.uniform(2.0, 4.5))
        mol_wt = float(L * 110.0 + rng.normal(0.0, 50.0))
        rows.append(list(counts) + [L, scd, shd, net, sum_lambda, beads_pos, beads_neg, shan_ent, mol_wt])
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def _toy_multitask(n_train: int = 24, seed: int = 0):
    """
    Build a tiny untrained multitask GPR + the stats needed to feed
    `GlobalGPRSurrogate(normalization_stats=...)`.

    Returns
    -------
    bundle : dict
    stats : dict (normalization stats)
    raw_test_df : pd.DataFrame (held-out raw features to predict)
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    raw_train_df = _make_raw_features_df(n_train, seed=seed)
    X_train_np, stats = convert_and_normalize_features(raw_train_df.to_numpy(np.float32), train=True)
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(rng.standard_normal((n_train, 2)), dtype=torch.float32)

    lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    model = MultitaskGPRegressionModel(X_train, y_train, lik, num_tasks=2)
    model.eval()
    lik.eval()

    raw_test_df = _make_raw_features_df(7, seed=seed + 1000)
    return {"model": model, "likelihood": lik, "X_train": X_train, "y_train": y_train}, stats, raw_test_df


def _toy_singletask(n_train: int = 24, seed: int = 0):
    """Build two tiny untrained singletask GPRs + stats."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    raw_train_df = _make_raw_features_df(n_train, seed=seed)
    X_train_np, stats = convert_and_normalize_features(raw_train_df.to_numpy(np.float32), train=True)
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_obj1 = torch.tensor(rng.standard_normal(n_train), dtype=torch.float32)
    y_obj2 = torch.tensor(rng.standard_normal(n_train), dtype=torch.float32)

    models, likelihoods, y_trains = {}, {}, {}
    for label, y in (("obj1", y_obj1), ("obj2", y_obj2)):
        lik = gpytorch.likelihoods.GaussianLikelihood()
        m = GPRegressionModel(X_train, y, lik)
        m.eval()
        lik.eval()
        models[label] = m
        likelihoods[label] = lik
        y_trains[label] = y

    raw_test_df = _make_raw_features_df(5, seed=seed + 2000)
    return ({"models": models, "likelihoods": likelihoods, "X_train": X_train, "y_train": y_trains},
            stats, raw_test_df)


def _fake_cfg(model_type: str, obj1: str = "obj1", obj2: str = "obj2"):
    """ALConfig stand-in with just the attributes make_surrogate reads."""
    return types.SimpleNamespace(train_model_type=model_type, obj1=obj1, obj2=obj2)


def _manual_predict_multitask(model, raw_df: pd.DataFrame, stats: dict):
    """The pre-refactor path: caller normalizes, then queries the GP directly."""
    X_np = convert_and_normalize_features(raw_df.to_numpy(np.float32), train=False, stats=stats)
    X_t = torch.tensor(np.asarray(X_np, dtype=np.float32), dtype=torch.float32)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        post = model(X_t)
        return post.mean.cpu().numpy(), post.stddev.cpu().numpy()


# ---------- multitask round-trip ----------

def test_multitask_predict_pool_matches_manual_normalize_then_predict():
    """
    The surrogate's internal normalize+predict pipeline must match the
    pre-refactor caller-normalizes-then-queries-GP path. If not, the
    abstraction silently changes numbers downstream.
    """
    bundle, stats, raw_test_df = _toy_multitask()

    cfg = _fake_cfg("gpr_multitask")
    sur = make_surrogate(cfg, model_bundle=bundle, normalization_stats=stats)
    pool = sur.predict_pool(raw_test_df)

    mu_manual, sd_manual = _manual_predict_multitask(bundle["model"], raw_test_df, stats)

    np.testing.assert_allclose(pool.means, mu_manual, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(pool.stds, sd_manual, rtol=1e-6, atol=1e-6)


def test_multitask_sample_shape_and_dtype():
    """sample(n) must produce (n, B, 2) torch float32 — the MC EHVI consumer's contract."""
    bundle, stats, raw_test_df = _toy_multitask()
    sur = make_surrogate(_fake_cfg("gpr_multitask"), model_bundle=bundle, normalization_stats=stats)

    pool = sur.predict_pool(raw_test_df)
    n = 8
    samples = pool.sample(n)
    assert samples.shape == (n, len(raw_test_df), 2)
    assert samples.dtype == torch.float32


def test_multitask_sample_reproducibility_with_seed():
    """Same torch seed -> same draws. Surrogate must not introduce hidden RNG state."""
    bundle, stats, raw_test_df = _toy_multitask()
    sur = make_surrogate(_fake_cfg("gpr_multitask"), model_bundle=bundle, normalization_stats=stats)
    pool = sur.predict_pool(raw_test_df)

    torch.manual_seed(99)
    a = pool.sample(4).clone()
    torch.manual_seed(99)
    b = pool.sample(4).clone()
    assert torch.equal(a, b)


def test_multitask_supports_joint_sampling_flag():
    bundle, stats, _ = _toy_multitask()
    sur = make_surrogate(_fake_cfg("gpr_multitask"), model_bundle=bundle, normalization_stats=stats)
    assert sur.supports_joint_sampling is True


# ---------- singletask round-trip ----------

def test_singletask_predict_pool_matches_per_objective_posteriors():
    """Column ordering must respect (cfg.obj1, cfg.obj2) — not the dict iteration order."""
    bundle, stats, raw_test_df = _toy_singletask()
    cfg = _fake_cfg("gpr_singletask", obj1="obj1", obj2="obj2")

    sur = make_surrogate(cfg, model_bundle=bundle, normalization_stats=stats)
    pool = sur.predict_pool(raw_test_df)

    # Pre-refactor path: manually normalize, then call each model directly.
    X_np = convert_and_normalize_features(raw_test_df.to_numpy(np.float32), train=False, stats=stats)
    Xt = torch.tensor(np.asarray(X_np, dtype=np.float32), dtype=torch.float32)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        p1 = bundle["models"]["obj1"](Xt)
        p2 = bundle["models"]["obj2"](Xt)

    np.testing.assert_allclose(pool.means[:, 0], p1.mean.cpu().numpy().ravel(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(pool.means[:, 1], p2.mean.cpu().numpy().ravel(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(pool.stds[:, 0], p1.stddev.cpu().numpy().ravel(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(pool.stds[:, 1], p2.stddev.cpu().numpy().ravel(), rtol=1e-6, atol=1e-6)


def test_singletask_sample_raises_not_implemented():
    """Single-task GPR has no cross-objective covariance — joint sampling must error."""
    bundle, stats, raw_test_df = _toy_singletask()
    sur = make_surrogate(_fake_cfg("gpr_singletask"), model_bundle=bundle, normalization_stats=stats)
    pool = sur.predict_pool(raw_test_df)
    with pytest.raises(NotImplementedError):
        pool.sample(4)


def test_singletask_supports_joint_sampling_flag():
    bundle, stats, _ = _toy_singletask()
    sur = make_surrogate(_fake_cfg("gpr_singletask"), model_bundle=bundle, normalization_stats=stats)
    assert sur.supports_joint_sampling is False


# ---------- factory + type hygiene ----------

def test_make_surrogate_returns_global_gpr_for_known_modes():
    """Factory dispatches gpr_multitask + gpr_singletask to GlobalGPRSurrogate."""
    mt_bundle, mt_stats, _ = _toy_multitask()
    st_bundle, st_stats, _ = _toy_singletask()
    sur_mt = make_surrogate(_fake_cfg("gpr_multitask"), model_bundle=mt_bundle, normalization_stats=mt_stats)
    sur_st = make_surrogate(_fake_cfg("gpr_singletask"), model_bundle=st_bundle, normalization_stats=st_stats)
    assert isinstance(sur_mt, GlobalGPRSurrogate)
    assert isinstance(sur_st, GlobalGPRSurrogate)
    assert isinstance(sur_mt, Surrogate)


def test_make_surrogate_rejects_unknown_mode():
    bundle, stats, _ = _toy_multitask()
    with pytest.raises(ValueError):
        make_surrogate(_fake_cfg("not-a-real-mode"), model_bundle=bundle, normalization_stats=stats)


def test_make_surrogate_global_requires_stats():
    """Defense in depth — caller must explicitly thread normalization stats through."""
    bundle, _stats, _ = _toy_multitask()
    with pytest.raises(ValueError):
        make_surrogate(_fake_cfg("gpr_multitask"), model_bundle=bundle, normalization_stats=None)


def test_global_gpr_surrogate_rejects_bad_mode_kwarg():
    """Direct ctor path also validates the mode string."""
    with pytest.raises(ValueError):
        GlobalGPRSurrogate(mode="bogus", model_bundle={}, normalization_stats={}, obj1="o1", obj2="o2")


def test_predict_pool_returns_pool_posterior_instance():
    bundle, stats, raw_test_df = _toy_multitask()
    sur = make_surrogate(_fake_cfg("gpr_multitask"), model_bundle=bundle, normalization_stats=stats)
    pool = sur.predict_pool(raw_test_df)
    assert isinstance(pool, PoolPosterior)
