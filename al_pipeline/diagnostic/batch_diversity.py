"""Quantitative diversity metrics for a produced batch of candidates.

The acquisition sweep (``--acq_test``) selects one sequence per GA child, giving
a batch of ``ngen`` candidates that would go to simulation. A UMAP scatter shows
their spread visually, but is not a defensible portfolio-diversity claim. This
module provides two well-established, reference-free numbers computed directly on
the normalized feature matrix:

- **Vendi score** (Friedman & Dieng, 2023): ``exp`` of the Shannon entropy of the
  eigenvalues of a normalized similarity kernel. Interpretable as the *effective
  number of distinct candidates* in the batch, bounded in ``[1, N]``. A batch of
  ``N`` near-duplicates scores ~1; a maximally spread batch scores ~``N``. We also
  report ``vendi_norm = vendi / N`` (a 0-1 efficiency).
- **Mean pairwise distance**: the linear-average companion to Vendi. Cosine
  distance ``1 - cos`` in the (high-dimensional, scale-invariant) feature space;
  Euclidean in a low-dimensional UMAP embedding where cosine is meaningless.

The same functionals are computed in feature space (cosine kernel) and, by the
aggregator, in a shared UMAP space (RBF kernel on Euclidean distance). Comparing
the two across sweep methods — Spearman rank agreement — is what validates the
UMAP as a faithful visual of diversity: see ``utils/acq_diversity_report.py``.

Feature-column layout (29 cols, from ``convert_and_normalize_features``):
``0:20`` amino-acid frequencies (composition), ``20:29`` physicochemical.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

# Feature blocks — mirror the full vs physicochemical split used by the UMAP
# plots (utils/gen_umap.py projects features[:, 20:] as the physicochem view).
FEATURE_BLOCKS: Dict[str, slice] = {
    "overall": slice(None),
    "composition": slice(0, 20),
    "physicochem": slice(20, 29),
}

_EPS = 1e-12


def _l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    """L2-normalize each row; zero-norm rows are left as zeros."""
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < _EPS, 1.0, norms)
    return X / norms


def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """N x N cosine-similarity Gram matrix (rows L2-normalized), clipped to [-1, 1]."""
    Xn = _l2_normalize_rows(X)
    return np.clip(Xn @ Xn.T, -1.0, 1.0)


def rbf_kernel_matrix(X: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """N x N RBF kernel ``exp(-d^2 / (2 sigma^2))``.

    ``sigma`` defaults to the median of the nonzero pairwise Euclidean distances
    (the standard median heuristic), which makes the kernel scale-adaptive.
    """
    X = np.asarray(X, dtype=np.float64)
    sq = np.sum(X ** 2, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (X @ X.T), 0.0)
    if sigma is None:
        n = X.shape[0]
        if n < 2:
            sigma = 1.0
        else:
            iu = np.triu_indices(n, k=1)
            d = np.sqrt(d2[iu])
            d = d[d > _EPS]
            sigma = float(np.median(d)) if d.size else 1.0
            if sigma < _EPS:
                sigma = 1.0
    return np.exp(-d2 / (2.0 * sigma * sigma))


def _vendi_from_kernel(K: np.ndarray) -> float:
    """Vendi score (order q=1) from a PSD similarity kernel with unit diagonal.

    Eigenvalues of ``K / n`` form a probability distribution (trace = 1); the
    score is ``exp`` of their Shannon entropy.
    """
    n = K.shape[0]
    if n <= 1:
        return float(n)
    w = np.linalg.eigvalsh(K / n)
    w = np.clip(w, 0.0, None)
    total = w.sum()
    if total <= _EPS:
        return 1.0
    w = w / total
    w = w[w > _EPS]
    entropy = -np.sum(w * np.log(w))
    return float(np.exp(entropy))


def vendi_score(X: np.ndarray, kernel: str = "cosine", sigma: float | None = None) -> float:
    """Effective number of distinct rows in ``X`` (Vendi score, order q=1).

    Parameters
    ----------
    X : np.ndarray
        Batch feature matrix, shape (N, D).
    kernel : {'cosine', 'rbf'}
        Similarity kernel. 'cosine' for high-dimensional feature space,
        'rbf' for a low-dimensional embedding.
    sigma : float, optional
        RBF bandwidth; median heuristic if None. Ignored for 'cosine'.

    Returns
    -------
    float
        Vendi score in ``[1, N]`` (0 for an empty batch).
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
    if kernel == "cosine":
        K = cosine_similarity_matrix(X)
    elif kernel == "rbf":
        K = rbf_kernel_matrix(X, sigma=sigma)
    else:
        raise ValueError(f"Unknown kernel: {kernel!r} (expected 'cosine' or 'rbf')")
    return _vendi_from_kernel(K)


def mean_pairwise_distance(X: np.ndarray, metric: str = "cosine") -> float:
    """Mean over unordered pairs of a distance.

    Parameters
    ----------
    X : np.ndarray
        Batch feature matrix, shape (N, D).
    metric : {'cosine', 'euclidean'}
        'cosine' -> ``1 - cos``; 'euclidean' -> L2 distance.

    Returns
    -------
    float
        Mean pairwise distance (0.0 when N < 2).
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n < 2:
        return 0.0
    iu = np.triu_indices(n, k=1)
    if metric == "cosine":
        S = cosine_similarity_matrix(X)
        return float((1.0 - S[iu]).mean())
    if metric == "euclidean":
        sq = np.sum(X ** 2, axis=1)
        d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (X @ X.T), 0.0)
        return float(np.sqrt(d2[iu]).mean())
    raise ValueError(f"Unknown metric: {metric!r} (expected 'cosine' or 'euclidean')")


def diversity_report(
    X: np.ndarray,
    kernel: str = "cosine",
    metric: str = "cosine",
) -> Dict[str, float]:
    """Vendi score, its N-normalized form, and mean pairwise distance for ``X``."""
    X = np.asarray(X, dtype=np.float64)
    n = int(X.shape[0])
    vs = vendi_score(X, kernel=kernel)
    return {
        "n": n,
        "vendi": vs,
        "vendi_norm": (vs / n) if n > 0 else 0.0,
        "mean_distance": mean_pairwise_distance(X, metric=metric),
    }


def diversity_rows(
    X: np.ndarray,
    space: str,
    kernel: str = "cosine",
    metric: str = "cosine",
    blocks: Dict[str, slice] | None = None,
) -> List[Dict]:
    """Tidy per-block report rows for one feature matrix.

    Each row is ``{space, block, n, vendi, vendi_norm, mean_distance}``. In a
    low-dimensional embedding (e.g. UMAP) only the 'overall' block is meaningful,
    so pass ``blocks={'overall': slice(None)}``.
    """
    X = np.asarray(X, dtype=np.float64)
    if blocks is None:
        blocks = FEATURE_BLOCKS if X.ndim == 2 and X.shape[1] >= 29 else {"overall": slice(None)}
    rows: List[Dict] = []
    for name, sl in blocks.items():
        rep = diversity_report(X[:, sl], kernel=kernel, metric=metric)
        rows.append({"space": space, "block": name, **rep})
    return rows


# ----------------------------------------------------------------------
# cfg-coupled orchestration (heavy imports deferred; used by the acq job)
# ----------------------------------------------------------------------

def read_selected_batch_features(cfg, log=None) -> np.ndarray:
    """Featurize + normalize the selected batch (one seq per GA child).

    Reads the first line of each ``seq_child_{i}.txt`` (the acquisition-selected
    sequence for that child), the same rows ``generate_simulation_candidates``
    sends to simulation, and returns their normalized 29-column feature matrix.
    """
    import json

    from al_pipeline.featurization.sequence_featurizer import SequenceFeaturizer
    from al_pipeline.data_prep.data_loading import convert_and_normalize_features

    p = cfg.paths
    children_dir = p.ga_children_dir
    seqs: List[str] = []
    for seq_id in range(1, cfg.ngen + 1):
        child_file = children_dir / f"seq_child_{seq_id}.txt"
        if not child_file.exists():
            raise FileNotFoundError(f"Missing child file: {child_file}")
        lines = child_file.read_text().strip().splitlines()
        seq = lines[0].strip() if lines else ""
        if not seq:
            raise ValueError(f"Empty sequence in: {child_file}")
        seqs.append(seq)

    featurizer = SequenceFeaturizer(cfg.model.lower(), str(cfg.db_path))
    raw = np.vstack(
        [np.asarray(featurizer.featurize(s), dtype=np.float64) for s in seqs]
    )

    with open(p.norm_stats, "r") as f:
        norm_stats = json.load(f)
    Xn = convert_and_normalize_features(raw, train=False, stats=norm_stats)
    return np.asarray(Xn, dtype=np.float32)


def score_selected_batch(cfg, log=None):
    """Compute feature-space diversity of the selected batch and persist it.

    Writes two files under ``cfg.paths.diagnostic_dir``, keyed by the run tag so
    each sweep method (ehvi x exploration) gets its own:

    - ``batch_features_norm_{tag}.csv`` — the normalized batch matrix, consumed by
      the shared-UMAP aggregator for the UMAP-space score + conservation check.
    - ``batch_diversity_{tag}.csv`` — tidy feature-space report (overall +
      composition + physicochem blocks).

    Returns the report DataFrame (or None if the batch could not be read).
    """
    import pandas as pd

    p = cfg.paths
    try:
        Xn = read_selected_batch_features(cfg, log=log)
    except (FileNotFoundError, ValueError) as e:
        msg = f"Batch diversity: could not read selected batch ({e}); skipping."
        if log:
            log.warning(msg)
        else:
            print(msg)
        return None

    diag_dir = p.diagnostic_dir
    diag_dir.mkdir(parents=True, exist_ok=True)

    feats_path = diag_dir / f"batch_features_norm_{p.tag}.csv"
    pd.DataFrame(Xn).to_csv(feats_path, index=False)

    rows = diversity_rows(Xn, space="feature", kernel="cosine", metric="cosine")
    for r in rows:
        r.update(
            model=cfg.model,
            tag=p.tag,
            ehvi_variant=cfg.ehvi_variant,
            exploration_strategy=cfg.exploration_strategy,
        )
    df = pd.DataFrame(rows)
    out_path = diag_dir / f"batch_diversity_{p.tag}.csv"
    df.to_csv(out_path, index=False)

    if log:
        overall = next(r for r in rows if r["block"] == "overall")
        log.info(
            f"Batch diversity (feature space, N={overall['n']}): "
            f"Vendi={overall['vendi']:.3f} ({overall['vendi_norm']:.2f} of N), "
            f"mean cosine dist={overall['mean_distance']:.3f}. Wrote {out_path}"
        )
    return df
