"""Tests for al_pipeline.diagnostic.batch_diversity.

Known-answer anchors: for a cosine kernel the Vendi score is exactly 1 for a set
of collinear (near-duplicate) vectors and exactly N for an orthonormal set, and
increases monotonically as a batch spreads out. Mean pairwise distance has the
same endpoints (0 for duplicates, 1 for orthogonal cosine).
"""
from __future__ import annotations

import numpy as np
import pytest

from al_pipeline.diagnostic.batch_diversity import (
    cosine_similarity_matrix,
    diversity_report,
    diversity_rows,
    mean_pairwise_distance,
    rbf_kernel_matrix,
    vendi_score,
)


def test_vendi_identical_rows_is_one():
    # All rows collinear -> rank-1 Gram matrix -> effective count 1.
    X = np.tile(np.array([1.0, 2.0, 3.0]), (8, 1))
    assert vendi_score(X, kernel="cosine") == pytest.approx(1.0, abs=1e-6)


def test_vendi_orthonormal_rows_is_n():
    # Orthogonal rows -> identity Gram -> effective count == N.
    X = np.eye(6)
    assert vendi_score(X, kernel="cosine") == pytest.approx(6.0, abs=1e-6)


def test_vendi_bounds_and_monotonic_spread():
    rng = np.random.default_rng(0)
    n, d = 12, 29
    base = rng.standard_normal(d)
    # Tight cluster (all near `base`) vs spread-out batch.
    tight = base + 1e-3 * rng.standard_normal((n, d))
    spread = rng.standard_normal((n, d))
    vs_tight = vendi_score(tight, kernel="cosine")
    vs_spread = vendi_score(spread, kernel="cosine")
    assert 1.0 <= vs_tight <= n + 1e-6
    assert 1.0 <= vs_spread <= n + 1e-6
    assert vs_tight < vs_spread


def test_mean_cosine_distance_endpoints():
    dup = np.tile(np.array([1.0, 1.0]), (5, 1))
    assert mean_pairwise_distance(dup, metric="cosine") == pytest.approx(0.0, abs=1e-9)
    orth = np.eye(4)
    # every off-diagonal cosine is 0 -> distance 1
    assert mean_pairwise_distance(orth, metric="cosine") == pytest.approx(1.0, abs=1e-9)


def test_mean_euclidean_distance():
    X = np.array([[0.0, 0.0], [3.0, 4.0]])  # single pair, distance 5
    assert mean_pairwise_distance(X, metric="euclidean") == pytest.approx(5.0)


def test_cosine_matrix_properties():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((7, 5))
    S = cosine_similarity_matrix(X)
    assert S.shape == (7, 7)
    assert np.allclose(np.diag(S), 1.0)
    assert np.allclose(S, S.T)
    assert S.min() >= -1.0 - 1e-9 and S.max() <= 1.0 + 1e-9


def test_rbf_kernel_unit_diagonal_and_psd():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((10, 4))
    K = rbf_kernel_matrix(X)
    assert np.allclose(np.diag(K), 1.0)
    # PSD: eigenvalues non-negative (allow tiny numerical negatives)
    assert np.linalg.eigvalsh(K).min() > -1e-8


def test_vendi_rbf_in_bounds():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((9, 2))
    vs = vendi_score(X, kernel="rbf")
    assert 1.0 <= vs <= 9.0 + 1e-6


def test_single_and_empty_batch():
    assert vendi_score(np.zeros((1, 5))) == pytest.approx(1.0)
    assert vendi_score(np.zeros((0, 5))) == pytest.approx(0.0)
    assert mean_pairwise_distance(np.zeros((1, 5))) == pytest.approx(0.0)


def test_diversity_report_keys_and_norm():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((8, 29))
    rep = diversity_report(X)
    assert set(rep) == {"n", "vendi", "vendi_norm", "mean_distance"}
    assert rep["n"] == 8
    assert rep["vendi_norm"] == pytest.approx(rep["vendi"] / 8)


def test_diversity_rows_blocks_on_29col():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((10, 29))
    rows = diversity_rows(X, space="feature")
    blocks = {r["block"] for r in rows}
    assert blocks == {"overall", "composition", "physicochem"}
    for r in rows:
        assert r["space"] == "feature"


def test_diversity_rows_defaults_to_overall_when_low_dim():
    rng = np.random.default_rng(6)
    emb = rng.standard_normal((10, 2))
    rows = diversity_rows(emb, space="umap", kernel="rbf", metric="euclidean")
    assert [r["block"] for r in rows] == ["overall"]


def test_bad_kernel_and_metric_raise():
    X = np.eye(3)
    with pytest.raises(ValueError):
        vendi_score(X, kernel="nope")
    with pytest.raises(ValueError):
        mean_pairwise_distance(X, metric="nope")
