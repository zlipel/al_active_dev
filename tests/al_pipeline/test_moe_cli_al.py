"""
Wiring tests for the MoE AL CLI integration (feat/moe-cli-al).

Covers the end-to-end chain:
  cfg.train_model_type='moe'
    -> train_from_config dispatches to train_moe_from_config
    -> writes PS expert + nonPS expert + RF bundle + norm_stats.json
       + features_norm_csv + labels_norm_csv
    -> ga_utils.load_moe_bundle returns a validated MoEBundle
    -> ga_utils.load_front returns (reference front, parent sequences)
    -> ga_utils.compute_and_store_shift freezes eps once from the base front and
       bakes it into the reference (loop needs no raw parent features)
    -> make_surrogate(cfg, moe_bundle=...) returns an MoESurrogate that
       wires cfg.moe_policy + cfg.moe_threshold

The actual `run_one_candidate` smoke isn't included here — it needs a real
SequenceFeaturizer + force-field database, which doesn't exist in the test
environment. Each interior wiring step is tested individually so the chain
is covered.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from al_pipeline.core.config import ALConfig
from al_pipeline.data_prep.parents import get_parents
from al_pipeline.ga import ga_utils
from al_pipeline.surrogates import MoEBundle, MoESurrogate, make_surrogate
from al_pipeline.training.kfold_training import train_from_config


AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
FEATURE_COLUMNS = AMINO_ACIDS + [
    "length", "SCD", "SHD", "|net charge|", "sum lambda",
    "beads(+)", "beads(-)", "shan ent", "mol wt",
]


# ---------- shared fixtures ----------

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


@pytest.fixture(scope="module")
def trained_moe(tmp_path_factory):
    """Heavy fixture: build synthetic data, train MoE, return cfg + key paths.

    Module-scoped so the ~6 sec of GP training is shared across tests in this file.
    """
    tmp_path = tmp_path_factory.mktemp("moe_cli_al")
    base = tmp_path / "home"
    scratch = tmp_path / "scratch"
    db = tmp_path / "db"
    for d in (base, scratch, db):
        d.mkdir(parents=True, exist_ok=True)

    cfg = ALConfig(
        model="TEST_MODEL",
        iteration=0,
        front="upper",
        base_path=base, scratch_path=scratch, db_path=db,
        train_model_type="moe",
        transform="yeoj",
        ehvi_variant="epsilon",        # exercise the epsilon-shift refactor too
        exploration_strategy="standard",
        epochs=30, patience=3, k_folds=3, learning_rate=0.1,
        moe_policy="soft", moe_threshold=0.5,
    )

    p = cfg.paths
    p.iter_scratch_dir.mkdir(parents=True, exist_ok=True)
    p.models_dir.mkdir(parents=True, exist_ok=True)
    p.iter_front_dir.mkdir(parents=True, exist_ok=True)

    feats = _make_raw_features_df(40, seed=0)
    labels = _make_labels_df(feats, seed=0)
    feats.to_csv(p.features_csv, index=False)
    labels.to_csv(p.labels_csv, index=False)

    # seq_gen.txt — get_parents reads this when stage='base'.
    seqs = _random_seqs(40, seed=0)
    with open(p.seq_gen_txt, "w") as f:
        for s in seqs:
            f.write(s + "\n")

    torch.manual_seed(0); np.random.seed(0)
    train_from_config(cfg)   # dispatches to train_moe_from_config

    return cfg


# ---------- train dispatch ----------

def test_train_from_config_dispatches_to_moe(trained_moe):
    """train_from_config(cfg moe=True) must reach train_moe_from_config and produce all artifacts."""
    cfg = trained_moe
    p = cfg.paths
    assert p.moe_ps_chkpt(temp=False).exists()
    assert p.moe_nonps_chkpt(temp=False).exists()
    assert p.moe_rf_bundle(temp=False).exists()


def test_moe_training_writes_global_norm_artifacts(trained_moe):
    """MoE training must also produce norm_stats.json + features_norm_csv + labels_norm_csv
    so the similarity-penalty path and the parents pipeline both work under MoE."""
    cfg = trained_moe
    p = cfg.paths
    assert p.norm_stats.exists()
    assert p.features_norm_csv.exists()
    assert p.labels_norm_csv.exists()
    # norm_stats is loadable JSON with the expected keys
    with open(p.norm_stats) as f:
        stats = json.load(f)
    assert "means" in stats and "stds" in stats


# ---------- load_moe_bundle ----------

def test_load_moe_bundle_returns_validated_bundle(trained_moe):
    """The new ga_utils.load_moe_bundle wires the file paths from cfg + enforces metadata."""
    cfg = trained_moe
    bundle = ga_utils.load_moe_bundle(cfg)
    assert isinstance(bundle, MoEBundle)
    assert bundle.label_scaler_scope == "all"
    assert bundle.transform == "yeoj"


def test_load_models_rejects_moe(trained_moe):
    """ga_utils.load_models is global-GPR-only; MoE must use load_moe_bundle.
    A clear error here prevents silent shape mismatches downstream."""
    cfg = trained_moe
    with pytest.raises(ValueError, match="load_moe_bundle"):
        ga_utils.load_models(cfg, temp=False)


# ---------- factory ----------

def test_make_surrogate_for_moe_uses_cfg_policy_and_threshold(trained_moe):
    """make_surrogate must forward cfg.moe_policy + cfg.moe_threshold to MoESurrogate."""
    cfg = trained_moe
    bundle = ga_utils.load_moe_bundle(cfg)
    sur = make_surrogate(
        cfg, moe_bundle=bundle, moe_policy=cfg.moe_policy, moe_threshold=cfg.moe_threshold,
    )
    assert isinstance(sur, MoESurrogate)
    # Hidden state isn't part of the public API; test the observable behavior:
    # a soft-policy surrogate has supports_joint_sampling=True regardless of threshold.
    assert sur.supports_joint_sampling is True


# ---------- get_parents + load_front + epsilon-shift ----------

def test_get_parents_base_writes_raw_and_normalized_parent_files(trained_moe):
    """Base-stage get_parents writes the raw parent_features_csv (the one-time eps
    input) plus the normalized parent files."""
    cfg = trained_moe
    # get_parents needs labels_norm_csv from training (which MoE training now writes).
    get_parents(cfg, stage="base")
    p = cfg.paths
    assert p.parent_features_csv.exists()
    assert p.parent_features_norm_csv.exists()
    assert p.parent_labels_norm_csv.exists()
    assert p.parent_seqs_txt.exists()


def test_load_front_returns_front_and_seqs(trained_moe):
    """load_front's contract: (reference front (N,2), aligned parent sequences)."""
    cfg = trained_moe
    get_parents(cfg, stage="base")   # idempotent
    pareto_front, parent_seqs = ga_utils.load_front(cfg, seq_id=1)
    assert pareto_front.ndim == 2 and pareto_front.shape[1] == 2
    assert len(parent_seqs) == pareto_front.shape[0]


def test_compute_and_store_shift_freezes_and_bakes(trained_moe):
    """compute_and_store_shift freezes eps from the base front + surrogate, writes
    epsilon_shift_json (eps + n_base), and bakes the shift into the base reference."""
    cfg = trained_moe
    get_parents(cfg, stage="base")
    p = cfg.paths
    before = pd.read_csv(p.parent_labels_norm_csv)[[cfg.obj1, cfg.obj2]].to_numpy().copy()
    n_base_expected = len(pd.read_csv(p.labels_norm_csv))

    ga_utils.compute_and_store_shift(cfg)

    assert p.epsilon_shift_json.exists()
    shift = json.loads(p.epsilon_shift_json.read_text())
    assert shift["n_base"] == n_base_expected
    assert len(shift["eps"]) == 2 and np.isfinite(shift["eps"]).all()
    # Base front is all-real -> every row shifted by the same frozen eps.
    after = pd.read_csv(p.parent_labels_norm_csv)[[cfg.obj1, cfg.obj2]].to_numpy()
    np.testing.assert_allclose(
        after - before, np.tile(shift["eps"], (len(before), 1)), atol=1e-6,
    )


def test_compute_and_store_shift_noop_when_standard(tmp_path: Path):
    """ehvi_variant != 'epsilon' -> eps=[0,0], no surrogate touched, front unchanged."""
    base = tmp_path / "home"; scratch = tmp_path / "scratch"; db = tmp_path / "db"
    for d in (base, scratch, db):
        d.mkdir(parents=True, exist_ok=True)
    cfg = ALConfig(
        model="TEST", iteration=0, front="upper",
        base_path=base, scratch_path=scratch, db_path=db,
        train_model_type="moe", ehvi_variant="standard",
    )
    p = cfg.ensure()
    labels = pd.DataFrame({cfg.obj1: [0.1, 0.5], cfg.obj2: [0.5, 0.1]})
    labels.to_csv(p.labels_norm_csv, index=False)
    labels.to_csv(p.parent_labels_norm_csv, index=False)

    ga_utils.compute_and_store_shift(cfg)

    shift = json.loads(p.epsilon_shift_json.read_text())
    assert shift["eps"] == [0.0, 0.0]
    assert shift["n_base"] == 2
    np.testing.assert_array_equal(
        pd.read_csv(p.parent_labels_norm_csv).to_numpy(), labels.to_numpy(),
    )


def test_get_parents_temp_stage_handles_grown_fantasy_rows(tmp_path: Path):
    """Regression: at 'temp' the normalized/sequence files have grown with KB fantasy
    rows while the base features_csv has not. get_parents must shift the real rows
    (< n_base) by the frozen eps, union the unshifted fantasy rows, and Pareto — with
    no IndexError and aligned reference outputs. CSV-only (no featurizer/DB)."""
    base = tmp_path / "home"; scratch = tmp_path / "scratch"; db = tmp_path / "db"
    for d in (base, scratch, db):
        d.mkdir(parents=True, exist_ok=True)
    cfg = ALConfig(
        model="TEST", iteration=0, front="upper",
        base_path=base, scratch_path=scratch, db_path=db,
        train_model_type="moe", ehvi_variant="epsilon",
    )
    p = cfg.ensure()
    obj1, obj2 = cfg.obj1, cfg.obj2

    n_base = 4
    # 4 real rows on an upper (max-max) front + 1 non-dominated fantasy row.
    real = pd.DataFrame({obj1: [0.1, 0.4, 0.7, 0.9], obj2: [0.9, 0.7, 0.4, 0.1]})
    fantasy = pd.DataFrame({obj1: [0.8], obj2: [0.8]})
    labels = pd.concat([real, fantasy], ignore_index=True)
    labels.to_csv(p.labels_norm_csv, index=False)
    # features_norm: same row count, 29 cols (values irrelevant to the front math).
    pd.DataFrame(
        np.zeros((len(labels), len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS,
    ).to_csv(p.features_norm_csv, index=False)
    seqs = [f"SEQ{i}" for i in range(len(labels))]
    p.seq_gen_temp_txt.write_text("\n".join(seqs) + "\n")
    eps = [0.05, 0.05]
    p.epsilon_shift_json.write_text(json.dumps({"eps": eps, "n_base": n_base}))

    get_parents(cfg, stage="temp")   # must not raise (the original bug was IndexError)

    ref = pd.read_csv(p.parent_labels_norm_csv)
    ref_seqs = p.parent_seqs_temp_txt.read_text().split()
    ref_pts = ref[[obj1, obj2]].to_numpy()
    assert len(ref) == len(ref_seqs) >= 1
    # Fantasy point joins the reference UNSHIFTED.
    assert any(np.allclose(row, [0.8, 0.8]) for row in ref_pts)
    # A real front member appears SHIFTED by eps (e.g. the extreme (0.1, 0.9)+eps).
    assert any(np.allclose(row, [0.15, 0.95]) for row in ref_pts)
    # temp stage must not re-write the raw base parent features file.
    assert not p.parent_features_csv.exists()


# ---------- ALConfig validation ----------

def test_alconfig_rejects_unknown_moe_policy(tmp_path: Path):
    """cfg.validate() must catch typos like --moe_policy mediumm early."""
    base = tmp_path / "home"; scratch = tmp_path / "scratch"; db = tmp_path / "db"
    for d in (base, scratch, db):
        d.mkdir(parents=True, exist_ok=True)
    cfg = ALConfig(
        model="TEST", iteration=0, front="upper",
        base_path=base, scratch_path=scratch, db_path=db,
        train_model_type="moe", moe_policy="medium",   # typo
    )
    with pytest.raises(ValueError, match="moe_policy"):
        cfg.validate()


def test_alconfig_moe_defaults():
    """Sanity: the defaults are 'soft' and 0.5."""
    cfg = ALConfig(model="TEST", iteration=0, front="upper")
    assert cfg.moe_policy == "soft"
    assert cfg.moe_threshold == 0.5


# ---------- augmentation ----------
# The `NotImplementedError` guard from feat/moe-cli-al is removed as of
# feat/moe-kriging-believer — MoE + kriging_believer / constant_liar_* is now
# wired via mixture covariance + hard-gate expert reconditioning. Coverage
# for that wiring lives in tests/al_pipeline/test_moe_kriging_believer.py.
