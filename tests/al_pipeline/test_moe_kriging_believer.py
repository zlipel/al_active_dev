"""
Tests for kriging-believer + MoE augmentation (feat/moe-kriging-believer).

Levels:
  1. `MoEPoolPosterior.covariance` — the mixture cov formula (soft policy) must
     agree with a Monte Carlo estimate of the joint (obj1, obj2) covariance
     under Bernoulli component selection. Hard policy: cov matches the
     assigned expert wholesale.

  2. `_MultitaskPoolPosterior.covariance` — behavior-preserving check against
     the pre-refactor `augmentation.get_cand_stats` on the same GP.

  3. `GPRExpert` direct-tensor checkpoint round-trip — the temp checkpoints
     written by augmentation store `train_x_direct` / `train_y_direct` (no
     `original_indices` reindex). Round-tripping must give identical
     predictions on the same inputs.

  4. `augment()` end-to-end under `kriging_believer + MoE`:
     - artifacts written to temp paths
     - assigned expert's train tensors expanded by exactly one row
     - unassigned expert's train tensors unchanged
     - RF bundle at temp path is identical to base
     - the "temp" bundle loads back and predicts (round-trip)
"""
from __future__ import annotations

import pickle
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import gpytorch
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import PowerTransformer

from al_pipeline.core.config import ALConfig
from al_pipeline.data_prep.data_loading import (
    apply_feature_normalizer, convert_features, fit_feature_normalizer,
)
from al_pipeline.ga import augmentation
from al_pipeline.ga.ga_utils import load_moe_bundle
from al_pipeline.surrogates import (
    GPRExpert, MoEBundle, MoEPoolPosterior, MoESurrogate,
    build_rf_features, load_rf_bundle,
)
from al_pipeline.surrogates.gpr_global import _MultitaskPoolPosterior
from al_pipeline.training.kfold_training import train_from_config
from al_pipeline.training.ml_models import MultitaskGPRegressionModel


AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
FEATURE_COLUMNS = AMINO_ACIDS + [
    "length", "SCD", "SHD", "|net charge|", "sum lambda",
    "beads(+)", "beads(-)", "shan ent", "mol wt",
]
LABEL_COLUMNS = ["exp_density", "diff"]


# ---------- helpers ----------

def _make_raw_features_df(n: int, seed: int) -> pd.DataFrame:
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


def _make_labels_df(features_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(features_df)
    z = rng.standard_normal(n) + (rng.random(n) < 0.5).astype(float) * 2.0 - 1.0
    density = z + 0.3 * features_df["SCD"].to_numpy()
    exp_density = density + rng.normal(0.0, 0.1, n)
    diff = 1.0 + np.abs(features_df["sum lambda"].to_numpy()) + rng.uniform(0.0, 0.5, n)
    return pd.DataFrame({
        "generation":         np.zeros(n, dtype=int),
        "density":            density,
        "density_std":        np.abs(rng.normal(0.0, 0.05, n)),
        "exp_density":        exp_density,
        "exp_density_std":    np.abs(rng.normal(0.0, 0.05, n)),
        "diff":               diff,
        "diff_std":           np.abs(rng.normal(0.0, 0.05, n)),
    })


def _random_seqs(n: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    aa = np.array(AMINO_ACIDS)
    return ["".join(aa[rng.integers(0, 20, size=int(rng.integers(20, 50)))]) for _ in range(n)]


def _train_toy_expert(feats: pd.DataFrame, labels: pd.DataFrame, scaler1, scaler2, transform="yeoj", seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    return GPRExpert.train(
        features_raw_df=feats, labels_raw_df=labels, label_columns=LABEL_COLUMNS,
        transform=transform, scaler1=scaler1, scaler2=scaler2,
        feature_columns=FEATURE_COLUMNS, lr=0.1, epochs=30, patience=3,
    )


def _fit_shared_scalers(labels_df: pd.DataFrame):
    scaler1 = PowerTransformer(method="yeo-johnson", standardize=True)
    scaler2 = PowerTransformer(method="yeo-johnson", standardize=True)
    scaler1.fit(labels_df[["exp_density"]].to_numpy())
    scaler2.fit(labels_df[["diff"]].to_numpy())
    return scaler1, scaler2


def _stamp(expert, regime):
    expert.label_scaler_scope = "all"
    expert.model_name = "TEST"
    expert.iteration = 0
    expert.regime = regime
    return expert


# ---------- (1) MoE mixture covariance ----------

def test_moe_soft_mixture_covariance_matches_mc(_bench_moe_bundle):
    """
    Under soft policy, `MoEPoolPosterior.covariance` must match the joint
    cov of a per-draw Bernoulli mixture of the two expert posteriors — the
    same MC contract used by the existing soft_mixture_variance test but on
    the full (2, 2) cov.
    """
    bundle = _bench_moe_bundle
    sur = MoESurrogate(bundle, policy="soft")
    feats = _make_raw_features_df(3, seed=42)
    pool = sur.predict_pool(feats)

    torch.manual_seed(0)
    n = 12000
    samples = pool.sample(n).cpu().numpy()   # (n, B, 2)
    mc_cov = np.stack([np.cov(samples[:, i, :].T) for i in range(samples.shape[1])])
    # 15% tolerance — cross-cov MC estimates are noisier than the diagonal.
    np.testing.assert_allclose(pool.covariance, mc_cov, rtol=0.15, atol=0.05)


def test_moe_hard_covariance_is_the_assigned_expert_cov(_bench_moe_bundle):
    """
    Hard policy with threshold=0 => everything goes PS. cov must equal the
    PS expert's own per-candidate cov. Analogous check for threshold=1+eps.
    """
    bundle = _bench_moe_bundle
    feats = _make_raw_features_df(4, seed=100)

    # Snapshot each expert's own per-candidate cov.
    post_ps = bundle.ps_expert.posterior(feats)
    post_nps = bundle.nonps_expert.posterior(feats)
    tmp = MoEPoolPosterior(
        p_ps=np.zeros(len(feats)),  # ignored below (we set policy=hard)
        post_ps=post_ps, post_nonps=post_nps, policy="hard", threshold=0.0,
    )
    cov_ps_only = tmp._per_expert_covariance(post_ps)
    cov_nps_only = tmp._per_expert_covariance(post_nps)

    # threshold=0 -> p_ps>=0 always -> pick PS
    sur_ps = MoESurrogate(bundle, policy="hard", threshold=0.0)
    pool_ps = sur_ps.predict_pool(feats)
    np.testing.assert_allclose(pool_ps.covariance, cov_ps_only, rtol=1e-6, atol=1e-6)

    # threshold>1 -> p_ps<threshold always -> pick nonPS
    sur_nps = MoESurrogate(bundle, policy="hard", threshold=1.0 + 1e-9)
    pool_nps = sur_nps.predict_pool(feats)
    np.testing.assert_allclose(pool_nps.covariance, cov_nps_only, rtol=1e-6, atol=1e-6)


def test_moe_soft_stds_consistent_with_covariance(_bench_moe_bundle):
    """
    The acquisition uses two views of the same mixture uncertainty: the epsilon
    shift + pessimism *scaling* read `pool.stds`, while the pessimism *overlap*
    reads `pool.covariance` (full law of total covariance). They must stay the
    same statistic: `stds**2 == diag(covariance)`. Locks `soft_mixture_variance`
    (the per-objective law of total variance) to the diagonal of the (2, 2)
    law-of-total-covariance so a refactor can't silently desync shift/scaling
    from the overlap. Checked under soft AND the hard-gate switch.
    """
    bundle = _bench_moe_bundle
    feats = _make_raw_features_df(5, seed=7)

    for sur in (
        MoESurrogate(bundle, policy="soft"),
        MoESurrogate(bundle, policy="hard", threshold=0.5),
    ):
        pool = sur.predict_pool(feats)
        diag = np.diagonal(pool.covariance, axis1=1, axis2=2)   # (B, 2)
        np.testing.assert_allclose(pool.stds ** 2, diag, rtol=1e-6, atol=1e-8)


# ---------- (2) _MultitaskPoolPosterior.covariance behavior-preserving ----------

def test_multitask_covariance_matches_get_cand_stats():
    """
    Behavior-preservation: the new `_MultitaskPoolPosterior.covariance`
    must match the old `augmentation.get_cand_stats` on the same posterior.
    """
    torch.manual_seed(0)
    n_train = 20
    X = torch.randn(n_train, 5)
    y = torch.randn(n_train, 2)
    lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    model = MultitaskGPRegressionModel(X, y, lik, num_tasks=2)
    model.eval(); lik.eval()

    X_test = torch.randn(6, 5)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        post = model(X_test)
    pool = _MultitaskPoolPosterior(post)
    mu_new, cov_new = pool.means, pool.covariance

    # Legacy path
    mu_old, cov_old = augmentation.get_cand_stats(model, X_test)
    np.testing.assert_allclose(mu_new, mu_old, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(cov_new, cov_old, rtol=1e-5, atol=1e-6)


# ---------- (3) GPRExpert direct-tensor checkpoint round-trip ----------

def test_gpr_expert_direct_tensor_checkpoint_roundtrip():
    """
    Temp checkpoints store `train_x_direct` + `train_y_direct` (no CSV
    reindex) so synthesized augmentation children survive save/load. Verify
    the reloaded expert matches the original on the same test features.
    """
    feats = _make_raw_features_df(20, seed=1)
    labels = _make_labels_df(feats, seed=1)
    scaler1, scaler2 = _fit_shared_scalers(labels)
    expert = _train_toy_expert(feats, labels, scaler1, scaler2)
    _stamp(expert, "ps")

    test_feats = _make_raw_features_df(5, seed=11)
    before = expert.predict(test_feats)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ckpt_path = td / "expert.pt"
        # Note: features/labels files are IGNORED when direct tensors present.
        # We still pass placeholder paths — from_checkpoint checks first for
        # the direct tensor keys.
        feats_path = td / "feats.csv"; labels_path = td / "labels.csv"
        feats.to_csv(feats_path, index=False)   # placeholder
        labels.to_csv(labels_path, index=False)

        expert.save_checkpoint(
            str(ckpt_path), regime="ps", label_scaler_scope="all",
            original_indices=[],   # NOT used because direct tensors present
            model_name="TEST", iteration=0,
            train_x_direct=expert.model.train_inputs[0],
            train_y_direct=expert.model.train_targets,
        )
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        assert "train_x_direct" in ckpt and "train_y_direct" in ckpt
        reloaded = GPRExpert.from_checkpoint(ckpt, str(feats_path), str(labels_path))

    after = reloaded.predict(test_feats)
    for k in before:
        np.testing.assert_allclose(after[k], before[k], rtol=1e-5, atol=1e-6)


# ---------- (4) _reindex_expert: adds exactly one row + preserves other state ----------

def test_reindex_expert_appends_one_row():
    feats = _make_raw_features_df(15, seed=2)
    labels = _make_labels_df(feats, seed=2)
    scaler1, scaler2 = _fit_shared_scalers(labels)
    expert = _train_toy_expert(feats, labels, scaler1, scaler2)

    n_before = expert.model.train_inputs[0].shape[0]
    x0 = expert.model.train_inputs[0].detach().numpy().copy()   # existing dataset
    new_child = _make_raw_features_df(1, seed=200)
    new_child_z = np.array([0.5, -0.2])
    train_x, train_y = augmentation._reindex_expert(expert, new_child, new_child_z, lr=0.1)
    assert train_x.shape[0] == n_before + 1
    assert train_y.shape[0] == n_before + 1
    # The expert's internal tensors were mutated to the expanded set.
    assert expert.model.train_inputs[0].shape[0] == n_before + 1

    # (req 5) Reconditioning is on dataset + fantasy: the existing rows are the
    # untouched training set, the one appended row IS the fantasy child.
    np.testing.assert_allclose(train_x[:n_before].detach().numpy(), x0, rtol=1e-6, atol=1e-6)
    # The appended label is exactly the passed (believed) z-space fantasy label.
    np.testing.assert_allclose(train_y[-1].detach().numpy(), new_child_z, rtol=1e-6, atol=1e-6)
    # (req 6) The appended feature row is the child normalized through THIS
    # expert's FROZEN normalizer — same convert+apply the loop reuses.
    feat_conv = convert_features(new_child[expert.feature_columns])
    expected_x = apply_feature_normalizer(feat_conv, expert.feature_normalizer_stats).to_numpy().ravel()
    np.testing.assert_allclose(train_x[-1].detach().numpy(), expected_x, rtol=1e-5, atol=1e-6)


# ---------- (5) end-to-end augment() under kriging_believer + MoE ----------

@pytest.fixture(scope="module")
def _bench_moe_bundle(tmp_path_factory):
    """A tiny MoEBundle for the covariance tests. Not backed by disk."""
    feats = _make_raw_features_df(24, seed=0)
    labels = _make_labels_df(feats, seed=0)
    is_ps = (labels["density"] > 0).to_numpy().astype(int)
    scaler1, scaler2 = _fit_shared_scalers(labels)

    ps_ex = _train_toy_expert(feats[is_ps == 1].reset_index(drop=True),
                                labels[is_ps == 1].reset_index(drop=True), scaler1, scaler2)
    nps_ex = _train_toy_expert(feats[is_ps == 0].reset_index(drop=True),
                                 labels[is_ps == 0].reset_index(drop=True), scaler1, scaler2)
    _stamp(ps_ex, "ps"); _stamp(nps_ex, "nonps")

    # A tiny RF.
    from sklearn.ensemble import RandomForestClassifier
    X_rf, conv_cols = build_rf_features(feats, FEATURE_COLUMNS, None)
    rf = RandomForestClassifier(n_estimators=10, random_state=0)
    rf.fit(X_rf, is_ps)
    rf_bundle: dict[str, Any] = {
        "classifier": rf,
        "rf_raw_feature_columns": FEATURE_COLUMNS,
        "rf_converted_feature_columns": conv_cols,
        "transform": "yeoj",
        "label_scaler_scope": "all",
        "model_name": "TEST",
        "iter": 0,
        "threshold": 0.5,
        "random_state": 0,
        "ps_definition": "density > 0",
    }
    return MoEBundle.from_components(rf_bundle, ps_ex, nps_ex)


class _FakeFeaturizer:
    """Stand-in for SequenceFeaturizer with deterministic per-sequence rows."""
    def __init__(self, *args, **kwargs):
        self._counter = 0

    def featurize(self, seq: str) -> list[float]:
        # Encode sequence identity in a way that produces distinct feature rows.
        rng = np.random.default_rng(hash(seq) & 0xFFFFFFFF)
        L = len(seq)
        probs = rng.dirichlet(np.ones(20))
        counts = rng.multinomial(L, probs).astype(float)
        engineered = [
            float(L), float(rng.normal(0.0, 0.5)), float(rng.uniform(0.0, 2.5)),
            float(rng.integers(0, 10)), float(rng.uniform(0.0, 3.5)),
            float(rng.integers(0, max(1, L // 5))), float(rng.integers(0, max(1, L // 5))),
            float(rng.uniform(2.0, 4.5)), float(L * 110.0 + rng.normal(0.0, 50.0)),
        ]
        return list(counts) + engineered


@pytest.fixture
def _moe_iter_dir(tmp_path, monkeypatch):
    """Full iter-0 layout on disk, trained MoE, ready for augment()."""
    base = tmp_path / "home"; scratch = tmp_path / "scratch"; db = tmp_path / "db"
    for d in (base, scratch, db):
        d.mkdir(parents=True, exist_ok=True)

    cfg = ALConfig(
        model="TEST_MODEL", iteration=0, front="upper",
        base_path=base, scratch_path=scratch, db_path=db,
        train_model_type="moe", transform="yeoj",
        ehvi_variant="standard", exploration_strategy="kriging_believer",
        pessimism=False,
        epochs=30, patience=3, k_folds=3, learning_rate=0.1,
        moe_policy="soft", moe_threshold=0.5,
    )
    p = cfg.paths
    for d in (p.iter_scratch_dir, p.models_dir, p.iter_front_dir, p.ga_children_dir):
        d.mkdir(parents=True, exist_ok=True)

    feats = _make_raw_features_df(40, seed=0)
    labels = _make_labels_df(feats, seed=0)
    feats.to_csv(p.features_csv, index=False)
    labels.to_csv(p.labels_csv, index=False)
    seqs = _random_seqs(40, seed=0)
    with open(p.seq_gen_txt, "w") as f:
        for s in seqs:
            f.write(s + "\n")

    torch.manual_seed(0); np.random.seed(0)
    train_from_config(cfg)   # dispatches to train_moe_from_config

    # A synthesized child sequence written to the GA candidates file.
    child = "MKKLVAGGGWLYNTRQPPRDDEELLSK"
    with open(p.ga_children_dir / "seq_child_1.txt", "w") as f:
        f.write(child + "\n")

    # Patch out SequenceFeaturizer since we don't have a real FF db in tests.
    monkeypatch.setattr(augmentation.sf, "SequenceFeaturizer", _FakeFeaturizer)
    return cfg


def test_augment_kriging_believer_moe_writes_temp_artifacts(_moe_iter_dir):
    cfg = _moe_iter_dir
    # Baseline snapshot: base checkpoints exist.
    p = cfg.paths
    assert p.moe_ps_chkpt(temp=False).exists()
    assert p.moe_nonps_chkpt(temp=False).exists()

    # Snapshot base expert train-sizes.
    bundle_base = load_moe_bundle(cfg, temp=False)
    ps_n_before = bundle_base.ps_expert.model.train_inputs[0].shape[0]
    nps_n_before = bundle_base.nonps_expert.model.train_inputs[0].shape[0]

    augmentation.augment(cfg, seq_id=1, pessimism=False)

    # Temp artifacts exist.
    assert p.moe_ps_chkpt(temp=True).exists()
    assert p.moe_nonps_chkpt(temp=True).exists()
    assert p.moe_rf_bundle(temp=True).exists()
    # seq_gen_temp updated too.
    assert p.seq_gen_temp_txt.exists()

    # Reloaded temp bundle: assigned expert grew by 1, other unchanged.
    bundle_temp = load_moe_bundle(cfg, temp=True)
    ps_n_after = bundle_temp.ps_expert.model.train_inputs[0].shape[0]
    nps_n_after = bundle_temp.nonps_expert.model.train_inputs[0].shape[0]
    grew_ps = (ps_n_after == ps_n_before + 1) and (nps_n_after == nps_n_before)
    grew_nps = (nps_n_after == nps_n_before + 1) and (ps_n_after == ps_n_before)
    assert grew_ps or grew_nps, (
        f"Expected exactly one expert to grow by 1: ps {ps_n_before}->{ps_n_after}, "
        f"nonps {nps_n_before}->{nps_n_after}"
    )


def test_augment_kriging_believer_moe_freezes_rf(_moe_iter_dir):
    """Temp RF bundle must be a copy of the base — the gate is frozen during batch generation."""
    cfg = _moe_iter_dir
    augmentation.augment(cfg, seq_id=1, pessimism=False)
    base = load_rf_bundle(str(cfg.paths.moe_rf_bundle(temp=False)))
    temp = load_rf_bundle(str(cfg.paths.moe_rf_bundle(temp=True)))
    # Compare the classifier's learned attributes rather than object identity
    # (pickle/re-pickle changes the identity but not the fitted state).
    for attr in ("n_estimators", "max_depth", "min_samples_leaf", "n_classes_"):
        assert getattr(base["classifier"], attr, None) == getattr(temp["classifier"], attr, None)
    # Predictions should be identical.
    feats = _make_raw_features_df(5, seed=999)
    X_rf, _ = build_rf_features(feats, FEATURE_COLUMNS, base["rf_converted_feature_columns"])
    np.testing.assert_array_equal(
        base["classifier"].predict_proba(X_rf), temp["classifier"].predict_proba(X_rf),
    )


def test_augment_kriging_believer_moe_pessimism_no_crash(_moe_iter_dir):
    """
    The pessimism penalty formula needs the (2, 2) mixture covariance. Under
    MoE this comes from `MoEPoolPosterior.covariance` — smoke-check that
    passing `pessimism=True` doesn't crash + still writes temp artifacts.
    """
    cfg = _moe_iter_dir
    cfg_pess = replace(cfg, pessimism=True)
    # Need seq_id > 1 to actually apply pessimism (see augment(): only
    # applies when prior children exist). Run seq_id=1 first to seed.
    augmentation.augment(cfg_pess, seq_id=1, pessimism=True)
    # seq_id=2 uses temp bundle from seq_id=1 as input.
    child2 = "GGGWLYNTKKKKKKKPPRDDEELLSKAAA"
    with open(cfg.paths.ga_children_dir / "seq_child_2.txt", "w") as f:
        f.write(child2 + "\n")
    augmentation.augment(cfg_pess, seq_id=2, pessimism=True)

    # Second-pass temp files exist and load.
    bundle = load_moe_bundle(cfg, temp=True)
    assert bundle.label_scaler_scope == "all"


# ---------- (6) routing / loop-file / pessimism / frozen-normalization ----------

class _DetFeaturizer:
    """Cross-process-deterministic stand-in for SequenceFeaturizer.

    `_FakeFeaturizer` keys on builtin `hash()`, which is per-process randomized
    (PYTHONHASHSEED) — fine for structural assertions but not for value-sensitive
    ones (pessimism magnitude). This one seeds off `hashlib.md5`, so a given
    sequence maps to the same feature row in every process.
    """
    def __init__(self, *args, **kwargs):
        pass

    def featurize(self, seq: str) -> list[float]:
        import hashlib
        seed = int(hashlib.md5(seq.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        L = len(seq)
        probs = rng.dirichlet(np.ones(20))
        counts = rng.multinomial(L, probs).astype(float)
        engineered = [
            float(L), float(rng.normal(0.0, 0.5)), float(rng.uniform(0.0, 2.5)),
            float(rng.integers(0, 10)), float(rng.uniform(0.0, 3.5)),
            float(rng.integers(0, max(1, L // 5))), float(rng.integers(0, max(1, L // 5))),
            float(rng.uniform(2.0, 4.5)), float(L * 110.0 + rng.normal(0.0, 50.0)),
        ]
        return list(counts) + engineered


def _expert_sizes(bundle) -> tuple[int, int]:
    return (
        bundle.ps_expert.model.train_inputs[0].shape[0],
        bundle.nonps_expert.model.train_inputs[0].shape[0],
    )


@pytest.mark.parametrize("threshold,expect_regime", [(0.0, "ps"), (1.0 + 1e-9, "nonps")])
def test_augment_kb_moe_routes_child_to_gated_expert(_moe_iter_dir, threshold, expect_regime):
    """(req 2) In KB mode the child (fantasy point) is hard-gated by the RF and
    reconditioned into that expert only. Force the gate with an extreme threshold
    (0 -> always PS, >1 -> always nonPS) and assert exactly the gated expert grew."""
    cfg = replace(_moe_iter_dir, moe_threshold=threshold)
    ps_before, nps_before = _expert_sizes(load_moe_bundle(cfg, temp=False))

    augmentation.augment(cfg, seq_id=1, pessimism=False)

    ps_after, nps_after = _expert_sizes(load_moe_bundle(cfg, temp=True))
    if expect_regime == "ps":
        assert ps_after == ps_before + 1 and nps_after == nps_before
    else:
        assert nps_after == nps_before + 1 and ps_after == ps_before


def test_augment_kb_moe_loop_consumes_and_grows_temp_norm_files(_moe_iter_dir):
    """(req 3) The AL loop lives in the temp/norm files: each seq_id appends the
    child to features_norm_csv / labels_norm_csv / seq_gen_temp_txt and, for
    seq_id>1, reloads the *temp* bundle (not base). Two augments must therefore
    grow the norm files by 2 and the experts' combined train set by 2 (chained
    through temp), not by 1 (which would mean seq_id=2 re-read the base bundle)."""
    cfg = _moe_iter_dir
    p = cfg.paths

    n_lbl0 = len(pd.read_csv(p.labels_norm_csv))
    n_feat0 = len(pd.read_csv(p.features_norm_csv))
    n_seq0 = len(p.seq_gen_txt.read_text().split())
    ps0, nps0 = _expert_sizes(load_moe_bundle(cfg, temp=False))

    augmentation.augment(cfg, seq_id=1, pessimism=False)
    assert len(pd.read_csv(p.labels_norm_csv)) == n_lbl0 + 1
    assert len(pd.read_csv(p.features_norm_csv)) == n_feat0 + 1
    assert len(p.seq_gen_temp_txt.read_text().split()) == n_seq0 + 1

    child2 = "GGGWLYNTKKKKKKKPPRDDEELLSKAAA"
    (p.ga_children_dir / "seq_child_2.txt").write_text(child2 + "\n")
    augmentation.augment(cfg, seq_id=2, pessimism=False)

    # Norm/seq files grew again -> seq_id=2 read the *grown* files, not the base.
    assert len(pd.read_csv(p.labels_norm_csv)) == n_lbl0 + 2
    assert len(pd.read_csv(p.features_norm_csv)) == n_feat0 + 2
    assert len(p.seq_gen_temp_txt.read_text().split()) == n_seq0 + 2

    # Combined expert train set grew by exactly 2 -> seq_id=2 loaded the temp
    # bundle from seq_id=1 (base+1) and chained (+1). Base reload would give +1.
    ps2, nps2 = _expert_sizes(load_moe_bundle(cfg, temp=True))
    assert (ps2 + nps2) == (ps0 + nps0) + 2


def test_augment_kb_moe_pessimism_conservative_and_reconditions_on_believed_label(
    _moe_iter_dir, monkeypatch,
):
    """(req 1, guards Fix 2) With --pessimism the believed label is pushed to the
    conservative side of the plain posterior mean, the pessimistic value (not the
    optimistic mean) is what's written to labels_norm_csv, and the assigned expert
    is reconditioned on that same pessimistic label — matching the global path."""
    # Deterministic featurizer so the pessimism penalty is reproducible.
    monkeypatch.setattr(augmentation.sf, "SequenceFeaturizer", _DetFeaturizer)
    cfg = replace(_moe_iter_dir, pessimism=True)   # front="upper" (maximize)
    p = cfg.paths

    augmentation.augment(cfg, seq_id=1, pessimism=True)   # seeds; no penalty at seq_id=1
    sizes_after1 = _expert_sizes(load_moe_bundle(cfg, temp=True))

    child2 = "MKKLVAGGGWLYNTRQPPRDDEELLSKGG"
    (p.ga_children_dir / "seq_child_2.txt").write_text(child2 + "\n")
    augmentation.augment(cfg, seq_id=2, pessimism=True)

    believed = pd.read_csv(p.labels_norm_csv).iloc[-1][[cfg.obj1, cfg.obj2]].to_numpy()
    optimistic = pd.read_csv(p.labels_no_pessimism).iloc[-1][[cfg.obj1, cfg.obj2]].to_numpy()

    # Pessimism actually moved the label to the conservative (lower, for the
    # upper/maximize front) side of the plain mean.
    assert not np.allclose(believed, optimistic)
    assert np.all(believed <= optimistic + 1e-8)

    # The reconditioned expert's newly appended target is the PESSIMISTIC label
    # (== labels_norm_csv tail), NOT the optimistic mean (labels_no_pessimism).
    bundle2 = load_moe_bundle(cfg, temp=True)
    ps2, nps2 = _expert_sizes(bundle2)
    grew = bundle2.ps_expert if ps2 == sizes_after1[0] + 1 else bundle2.nonps_expert
    recond_y = grew.model.train_targets[-1].detach().numpy()
    np.testing.assert_allclose(recond_y, believed, rtol=1e-5, atol=1e-6)
    assert not np.allclose(recond_y, optimistic)


def test_augment_kb_moe_freezes_and_reuses_normalization_on_fantasy(_moe_iter_dir):
    """(req 6) The per-expert feature normalizer + label scalers and the global
    norm_stats.json are frozen during the batch and simply *reused* to encode the
    fantasy child: the reconditioned feature row equals the child normalized with
    the base (pre-augment) frozen normalizer, and no normalization state changes."""
    cfg = _moe_iter_dir
    p = cfg.paths

    base_bundle = load_moe_bundle(cfg, temp=False)
    ps0, nps0 = _expert_sizes(base_bundle)
    # Snapshot frozen normalization state before the batch touches anything.
    norm_stats_before = p.norm_stats.read_bytes()
    ps_feat_stats_before = base_bundle.ps_expert.feature_normalizer_stats
    nps_feat_stats_before = base_bundle.nonps_expert.feature_normalizer_stats

    augmentation.augment(cfg, seq_id=1, pessimism=False)

    # Global normalization stats untouched by the loop.
    assert p.norm_stats.read_bytes() == norm_stats_before

    temp_bundle = load_moe_bundle(cfg, temp=True)
    ps1, nps1 = _expert_sizes(temp_bundle)
    grew_is_ps = ps1 == ps0 + 1
    grew_before = base_bundle.ps_expert if grew_is_ps else base_bundle.nonps_expert
    grew_after = temp_bundle.ps_expert if grew_is_ps else temp_bundle.nonps_expert

    # Per-expert feature normalizer is frozen through the batch. Stats are nested
    # {feature: {'type', 'mean'/'std'/'min'/...}} — walk and compare numerically.
    frozen_stats = ps_feat_stats_before if grew_is_ps else nps_feat_stats_before
    after_stats = grew_after.feature_normalizer_stats
    assert set(after_stats) == set(frozen_stats)
    for feat, sub in frozen_stats.items():
        assert after_stats[feat]["type"] == sub["type"]
        for stat_key, val in sub.items():
            if stat_key == "type":
                continue
            np.testing.assert_allclose(
                float(after_stats[feat][stat_key]), float(val), rtol=1e-6, atol=1e-8,
            )
    # Label scalers frozen too (Yeo-Johnson lambdas unchanged).
    np.testing.assert_allclose(
        grew_after.label_scaler1.lambdas_, grew_before.label_scaler1.lambdas_, rtol=1e-6,
    )
    np.testing.assert_allclose(
        grew_after.label_scaler2.lambdas_, grew_before.label_scaler2.lambdas_, rtol=1e-6,
    )

    # The fantasy child's reconditioned feature row is the child encoded with the
    # FROZEN (base) normalizer — normalization was reused, not refit.
    child = (cfg.paths.ga_children_dir / "seq_child_1.txt").read_text().strip()
    featurizer = augmentation.sf.SequenceFeaturizer(cfg.model.lower(), cfg.db_path)
    raw = pd.DataFrame(
        np.asarray(featurizer.featurize(child)).reshape(1, -1), columns=FEATURE_COLUMNS,
    )
    feat_conv = convert_features(raw[grew_before.feature_columns])
    expected_x = apply_feature_normalizer(
        feat_conv, grew_before.feature_normalizer_stats,
    ).to_numpy().ravel()
    np.testing.assert_allclose(
        grew_after.model.train_inputs[0][-1].detach().numpy(), expected_x, rtol=1e-5, atol=1e-6,
    )


def test_augment_kb_moe_pessimism_feeds_raw_prev_children(_moe_iter_dir, monkeypatch):
    """Regression: the pessimism overlap must RE-FEATURIZE prior children (RAW),
    not reuse the globally-normalized rows from features_norm_csv. Feeding
    normalized rows as "raw" sends the MoE RF gate's convert_features to inf
    ('Input X contains infinity ...'). Spy the surrogate calls and assert the
    prior-children frame carries RAW features, not the normalized ones."""
    monkeypatch.setattr(augmentation.sf, "SequenceFeaturizer", _DetFeaturizer)
    cfg = replace(_moe_iter_dir, pessimism=True)
    p = cfg.paths

    calls: list[pd.DataFrame] = []
    orig = augmentation.predict_for_augmentation

    def _spy(surrogate, features_raw_df, return_std=False):
        calls.append(features_raw_df.copy())
        return orig(surrogate, features_raw_df, return_std=return_std)

    monkeypatch.setattr(augmentation, "predict_for_augmentation", _spy)

    augmentation.augment(cfg, seq_id=1, pessimism=True)   # seeds child_1
    child2 = "MKKLVAGGGWLYNTRQPPRDDEELLSKGG"
    (p.ga_children_dir / "seq_child_2.txt").write_text(child2 + "\n")
    calls.clear()
    augmentation.augment(cfg, seq_id=2, pessimism=True)   # would crash under the bug

    # Two predict calls at seq_id=2: [current child, prior children].
    assert len(calls) == 2
    prev_df = calls[1]
    assert len(prev_df) == 1                              # one prior child (child_1)
    assert np.isfinite(prev_df.to_numpy()).all()          # not inf/over-range

    # The prior-children row is the RAW featurization of child_1, NOT its
    # normalized row in features_norm_csv (which is what the bug fed).
    child1 = (p.ga_children_dir / "seq_child_1.txt").read_text().strip()
    expected_raw = np.asarray(_DetFeaturizer().featurize(child1), dtype=float)
    np.testing.assert_allclose(prev_df.iloc[0].to_numpy(), expected_raw, rtol=1e-6, atol=1e-6)
    norm_row = pd.read_csv(p.features_norm_csv).iloc[-2].to_numpy()   # child_1's normalized row
    assert not np.allclose(prev_df.iloc[0].to_numpy(), norm_row)
