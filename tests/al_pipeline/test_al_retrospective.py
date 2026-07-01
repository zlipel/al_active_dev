"""
Tests for `al_pipeline.diagnostic.al_retrospective`.

Coverage:
  1. `IterationData` slicing (training-before + proposal-at) is index-consistent
     across features/labels/seqs.
  2. `compute_target_front` matches `data_prep.parents.find_pareto_front` on
     the union of iters.
  3. `score_children_ehvi` for a fake `PoolPosterior` recovers the same
     ranking as calling `ehvi_analytic` directly.
  4. `run_retrospective` end-to-end on tiny synthetic data:
     - writes summary CSV + trajectory JSON with expected schema,
     - HV trajectories are monotonic non-decreasing (Pareto only grows),
     - target HV >= max policy HV (target is the union of all iters).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from al_pipeline.acquisition import ehvi
from al_pipeline.core.config import ALConfig
from al_pipeline.data_prep.parents import find_pareto_front
from al_pipeline.diagnostic.al_retrospective import (
    IterationData,
    compute_hv_raw,
    compute_target_front,
    load_completed_run,
    run_retrospective,
    score_children_ehvi,
    _global_ref_point,
)
from al_pipeline.surrogates.base import PoolPosterior


AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
FEATURE_COLUMNS = AMINO_ACIDS + [
    "length", "SCD", "SHD", "|net charge|", "sum lambda",
    "beads(+)", "beads(-)", "shan ent", "mol wt",
]


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


def _make_labels_df(features_df: pd.DataFrame, generation_seq: np.ndarray, seed: int) -> pd.DataFrame:
    """Synthetic labels with a `generation` column that identifies which iter each row belongs to."""
    rng = np.random.default_rng(seed)
    n = len(features_df)
    z = rng.standard_normal(n) + (rng.random(n) < 0.5).astype(float) * 2.0 - 1.0
    density = z + 0.3 * features_df["SCD"].to_numpy()
    exp_density = density + rng.normal(0.0, 0.1, n)
    diff = 1.0 + np.abs(features_df["sum lambda"].to_numpy()) + rng.uniform(0.0, 0.5, n)
    return pd.DataFrame({
        "generation":         generation_seq.astype(int),
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


# ---------- (1) IterationData slicing ----------

def test_iteration_data_slicing_index_consistent():
    """training_slice_before(N) and proposal_pool_at(N) must return rows that
    exist in the underlying features/labels/seqs, with lengths that match."""
    feats = _make_raw_features_df(30, seed=0)
    gens = np.concatenate([np.zeros(10), np.ones(10), np.full(10, 2)]).astype(int)
    labels = _make_labels_df(feats, gens, seed=0)
    seqs = _random_seqs(30, seed=0)

    data = IterationData(features_df=feats, labels_df=labels, seqs=seqs)

    train_f, train_l, train_s = data.training_slice_before(2)
    assert len(train_f) == len(train_l) == len(train_s) == 20   # gens 0 + 1
    assert set(train_l["generation"].unique()) == {0, 1}

    pool_f, pool_l, pool_s = data.proposal_pool_at(2)
    assert len(pool_f) == len(pool_l) == len(pool_s) == 10
    assert set(pool_l["generation"].unique()) == {2}


def test_iteration_data_rejects_mis_aligned_input():
    feats = _make_raw_features_df(5, seed=1)
    labels = _make_labels_df(feats, np.zeros(5), seed=1).iloc[:3]   # too short
    with pytest.raises(ValueError, match="mis-aligned"):
        IterationData(features_df=feats, labels_df=labels, seqs=["a"] * 5)


def test_iteration_data_requires_generation_column():
    feats = _make_raw_features_df(3, seed=1)
    labels = pd.DataFrame({"exp_density": [1.0, 2.0, 3.0], "diff": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="generation"):
        IterationData(features_df=feats, labels_df=labels, seqs=["a", "b", "c"])


# ---------- (2) target front vs. find_pareto_front ----------

def test_compute_target_front_matches_find_pareto_front():
    feats = _make_raw_features_df(30, seed=2)
    gens = np.concatenate([np.zeros(15), np.ones(15)]).astype(int)
    labels = _make_labels_df(feats, gens, seed=2)
    target = compute_target_front(labels, front="upper", obj1="exp_density", obj2="diff")

    # Direct call to find_pareto_front on the same data
    front_df, _idx = find_pareto_front(
        labels[["exp_density", "diff"]], kind=["max", "max"], objectives=["exp_density", "diff"],
    )
    np.testing.assert_allclose(target, front_df[["exp_density", "diff"]].to_numpy())


# ---------- (3) score_children_ehvi vs. ehvi_analytic directly ----------

class _FakePoolPosterior(PoolPosterior):
    """Deterministic mean+std; independent per-obj cov diagonal."""
    def __init__(self, means: np.ndarray, stds: np.ndarray):
        self._means = np.asarray(means, dtype=np.float64)
        self._stds = np.asarray(stds, dtype=np.float64)

    @property
    def means(self):
        return self._means

    @property
    def stds(self):
        return self._stds

    @property
    def covariance(self):
        B, D = self._stds.shape
        cov = np.zeros((B, D, D))
        for t in range(D):
            cov[:, t, t] = self._stds[:, t] ** 2
        return cov

    def sample(self, n_samples: int):
        raise NotImplementedError


class _FakeSurrogate:
    def __init__(self, means, stds):
        self._pool = _FakePoolPosterior(means, stds)

    def predict_pool(self, X_raw):
        return self._pool


def test_score_children_ehvi_matches_direct_ehvi_analytic():
    """The scorer should equal a direct ehvi_analytic call for the same posterior."""
    means = np.array([[1.0, 1.0], [2.0, 0.5], [0.5, 2.0]], dtype=np.float64)
    stds = np.full_like(means, 0.3)
    surrogate = _FakeSurrogate(means, stds)

    pareto_front = np.array([[1.5, 1.5], [2.5, 1.0], [1.0, 2.5]], dtype=np.float64)
    dummy_feats = pd.DataFrame(np.zeros((3, 5)), columns=list("abcde"))

    got = score_children_ehvi(
        surrogate, dummy_feats, pareto_front, front="lower",
        ref_mode="frac", ref_frac=0.5,
    )
    aug = ehvi.front_augmentation(pareto_front, front="lower", ref_mode="frac", frac=0.5)
    want = ehvi.ehvi_analytic(means[:, 0], stds[:, 0], means[:, 1], stds[:, 1], aug)
    np.testing.assert_allclose(got, want, rtol=1e-8, atol=1e-10)


# ---------- (4) compute_hv_raw sanity ----------

def test_compute_hv_raw_monotone_under_added_pareto_point():
    """Adding a dominating point cannot decrease HV."""
    labels_a = pd.DataFrame({"exp_density": [1.0, 2.0], "diff": [2.0, 1.0]})
    labels_b = pd.concat([labels_a, pd.DataFrame({"exp_density": [3.0], "diff": [3.0]})])
    ref_pt = _global_ref_point(labels_b, front="upper", obj1="exp_density", obj2="diff")
    hv_a = compute_hv_raw(labels_a, front="upper", obj1="exp_density", obj2="diff", ref_point_min=ref_pt)
    hv_b = compute_hv_raw(labels_b, front="upper", obj1="exp_density", obj2="diff", ref_point_min=ref_pt)
    assert hv_b >= hv_a


# ---------- (5) end-to-end run_retrospective on synthetic data ----------

def _write_completed_run(tmp_path: Path, n_iters: int, batch_size: int, seed: int) -> Path:
    """
    Materialize a synthetic completed AL run at the layout the loader expects:
       runs_root / <MODEL> / GENERATIONS / iteration_N /
           features_gen{N}.csv, labels_gen{N}.csv, seq_gen{N}.txt
    Only the FINAL iter's cumulative files are needed by load_completed_run.
    """
    runs_root = tmp_path / "runs"
    model = "TEST_MODEL"
    gen_dir = runs_root / model / "GENERATIONS" / f"iteration_{n_iters}"
    gen_dir.mkdir(parents=True, exist_ok=True)

    n_seed = 40
    total = n_seed + n_iters * batch_size
    feats = _make_raw_features_df(total, seed=seed)
    # generation vector: 40 rows at gen=0 (seed), then batch_size rows per subsequent iter
    gens = np.concatenate([
        np.zeros(n_seed),
        *[np.full(batch_size, i) for i in range(1, n_iters + 1)],
    ]).astype(int)
    labels = _make_labels_df(feats, gens, seed=seed + 1)
    seqs = _random_seqs(total, seed=seed + 2)

    feats.to_csv(gen_dir / f"features_gen{n_iters}.csv", index=False)
    labels.to_csv(gen_dir / f"labels_gen{n_iters}.csv", index=False)
    with open(gen_dir / f"seq_gen{n_iters}.txt", "w") as f:
        for s in seqs:
            f.write(s + "\n")
    return runs_root


@pytest.mark.slow
def test_run_retrospective_end_to_end_synthetic(tmp_path):
    """
    Tiny 3-iter synthetic run. Verifies:
      - both output files exist with the expected schema
      - HV trajectories are monotonic non-decreasing
      - target HV >= any policy's HV (target is union of all iters)
    """
    n_iters = 3
    batch = 12
    runs_root = _write_completed_run(tmp_path, n_iters=n_iters, batch_size=batch, seed=7)

    # cfg_base: diagnostic_dir will land at base_path/<MODEL>/DIAGNOSTIC.
    base = tmp_path / "home"
    scratch = tmp_path / "scratch"
    db = tmp_path / "db"
    for d in (base, scratch, db):
        d.mkdir(parents=True, exist_ok=True)
    cfg_base = ALConfig(
        model="TEST_MODEL", iteration=0, front="upper",
        base_path=base, scratch_path=scratch, db_path=db,
        train_model_type="moe", transform="yeoj",
        ehvi_variant="epsilon", exploration_strategy="standard",
        obj1="exp_density", obj2="diff",
        epochs=20, patience=3, k_folds=3, learning_rate=0.1,
        ngen=batch,
        moe_policy="soft", moe_threshold=0.5,
    )

    torch.manual_seed(0); np.random.seed(0)
    out = run_retrospective(
        runs_root=runs_root, model="TEST_MODEL",
        cfg_base=cfg_base, n_iters=n_iters,
        k_pick=batch // 2,
    )

    diag_dir = cfg_base.paths.diagnostic_dir
    assert (diag_dir / "retrospective_summary.csv").exists()
    assert (diag_dir / "retrospective_trajectory.json").exists()

    summary = pd.read_csv(diag_dir / "retrospective_summary.csv")
    assert len(summary) == n_iters
    for col in ("iter", "n_children", "n_picked",
                 "hv_actual", "hv_moe_soft", "hv_moe_hard", "hv_global"):
        assert col in summary.columns, f"missing column {col}"

    with open(diag_dir / "retrospective_trajectory.json") as f:
        traj = json.load(f)
    for key in ("iters", "hv_actual", "hv_moe_soft", "hv_moe_hard", "hv_global"):
        assert key in traj
        assert len(traj[key]) == n_iters

    # Monotone non-decreasing under each policy.
    for name in ("hv_actual", "hv_moe_soft", "hv_moe_hard", "hv_global"):
        vals = np.array(traj[name])
        assert np.all(np.diff(vals) >= -1e-9), f"{name} not monotone: {vals}"

    # Target HV bounds every policy's final HV.
    for name in ("hv_actual", "hv_moe_soft", "hv_moe_hard", "hv_global"):
        assert traj[name][-1] <= traj["target_hv"] + 1e-8, \
            f"{name} final HV exceeds target HV: {traj[name][-1]} > {traj['target_hv']}"


# ---------- load_completed_run error paths ----------

def test_load_completed_run_missing_file_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing completed-run artifact"):
        load_completed_run(tmp_path, model="NO_SUCH_MODEL", n_iters=1)
