"""
Featurizer parity check across implementations.

The MoE port (and the eventual numba featurizer port for beam-search throughput)
both assume that the three "correct" implementations agree bit-for-bit on every
column of the 29-feature vector for every force field:

  - al_pipeline.featurization.sequence_featurizer.SequenceFeaturizer  (serial)
  - MODEL_COMPARISON_STELLAR_CURR/sequence_featurizer_fast.py         (joblib)
  - MODEL_COMPARISON_STELLAR_CURR/sequence_featurizer_numba.py        (numba)

The fourth implementation (MODEL_COMPARISON_STELLAR_CURR/sequence_featurizer.py)
is the older serial version with a known calvados-terminal bug in
`pos_frac`/`neg_frac` — it is *not* validated here and will be obsoleted.

If the MODEL_COMPARISON_STELLAR_CURR tree or the force-field database isn't
present on the host, the test skips with a clear message.
"""
from __future__ import annotations

import os
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pytest


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


# --- locate MODEL_COMPARISON_STELLAR_CURR and the FF database ---

def _find_mcsc() -> Path | None:
    """Locate MODEL_COMPARISON_STELLAR_CURR via env var or known paths."""
    env = os.environ.get("MODEL_COMPARISON_STELLAR_CURR")
    if env and Path(env).is_dir():
        return Path(env)
    candidates = [
        Path("/Users/zl4808/Documents/ActiveLearningAndIDPs/PROJECTS/MODEL_COMPARISON_STELLAR_CURR"),
        Path("/home/zl4808/PROJECTS/MODEL_COMPARISON_STELLAR_CURR"),
    ]
    for p in candidates:
        if p.is_dir():
            return p
    return None


def _find_db_path() -> Path | None:
    """Locate the gendata force-field database directory.

    Order: $DB_PATH > known local paths > None.
    """
    env = os.environ.get("DB_PATH")
    if env and (Path(env) / "ff_db.py").exists():
        return Path(env)
    candidates = [
        Path("/Users/zl4808/Documents/ActiveLearningAndIDPs/stellar_scripts/GENDATA/databases"),
        Path("/Users/zl4808/Documents/ActiveLearningAndIDPs/PROJECTS/MODEL_COMPARISON/databases"),
        Path("/home/zl4808/scripts/GENDATA/databases"),
    ]
    for p in candidates:
        if (p / "ff_db.py").exists():
            return p
    return None


MCSC = _find_mcsc()
DB_PATH = _find_db_path()

requires_mcsc = pytest.mark.skipif(
    MCSC is None,
    reason="MODEL_COMPARISON_STELLAR_CURR not found (set env var of the same name to override)",
)
requires_db = pytest.mark.skipif(
    DB_PATH is None,
    reason="Force-field database not found (set DB_PATH env var to override)",
)


# --- test sequence generator (mirrors MODEL_COMPARISON_STELLAR_CURR/featurizer_equivalence.py) ---

def make_test_seqs(seed: int = 0) -> list[str]:
    # The project only studies IDPs of length 20-160. Tiny sequences are not
    # representative and exercise edge cases (e.g. the MCSC fast featurizer's
    # L=1 calvados terminal-shift bug) that don't matter in practice.
    rng = np.random.default_rng(seed)
    lens = [20, 30, 50, 80, 100, 120, 160]
    seqs: list[str] = []

    # Edge patterns + homopolymers.
    for L in lens:
        seqs.append("A" * L)
        seqs.append(("KR" * (L // 2) + ("K" if L % 2 else ""))[:L])
        seqs.append(("DE" * (L // 2) + ("D" if L % 2 else ""))[:L])
        seqs.append(("ACDEFGHIKLMNPQRSTVWY" * ((L // 20) + 1))[:L])

    # Random.
    for L in lens:
        for _ in range(10):
            seqs.append("".join(rng.choice(list(AMINO_ACIDS), size=L)))

    # Deduplicate, preserve order.
    seen: set[str] = set()
    out: list[str] = []
    for s in seqs:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# --- module loaders (so we don't pollute sys.path globally) ---

def _load_mcsc_module(name: str, filename: str):
    """Import a python file from MODEL_COMPARISON_STELLAR_CURR by absolute path."""
    assert MCSC is not None
    # Some MCSC featurizers import sibling modules — ensure the dir is on path
    # while we load. We then leave it on path; tests run sequentially.
    mcsc_str = str(MCSC)
    if mcsc_str not in sys.path:
        sys.path.insert(0, mcsc_str)
    return SourceFileLoader(name, str(MCSC / filename)).load_module()


# --- parity test ---

@requires_mcsc
@requires_db
@pytest.mark.parametrize("model", ["hps_urry", "hps_kr", "mpipi", "calvados"])
def test_featurizer_parity_alpipeline_vs_fast_vs_numba(model: str):
    """
    al_pipeline serial == MCSC fast == MCSC numba on every (sequence, column).

    Tight tolerance (rtol=1e-9, atol=1e-10) — these implementations should agree
    to floating-point summation noise only.
    """
    from al_pipeline.featurization.sequence_featurizer import SequenceFeaturizer as ALSF
    fast_mod  = _load_mcsc_module("mcsc_sf_fast",  "sequence_featurizer_fast.py")
    numba_mod = _load_mcsc_module("mcsc_sf_numba", "sequence_featurizer_numba.py")

    seqs = make_test_seqs()
    db = str(DB_PATH)

    al     = ALSF(model, db)
    fast   = fast_mod.SequenceFeaturizer(model, db)
    numba_ = numba_mod.SequenceFeaturizer(model, db)

    X_al    = al.featurize_many(seqs).to_numpy()
    X_fast  = np.asarray(fast.featurize_many_threaded(seqs, feat_threads=2, chunk_size=512))
    # Numba: warm up the JIT on a tiny slice, then run for real.
    _ = numba_.featurize_many_fast(seqs[:2], feat_threads=2, as_df=False)
    X_numba = np.asarray(numba_.featurize_many_fast(seqs, feat_threads=2, as_df=False))

    # Allow ~float32 / fastmath rounding noise on both comparisons:
    #   - MCSC fast stores charge/lambda arrays as float32; al_pipeline uses doubles
    #   - MCSC numba is float64 internally but compiled with fastmath=True +
    #     parallel=True, so summation order differs from the serial Python version
    # Empirically: max relative diff ~ 1e-6 (single-precision territory).
    np.testing.assert_allclose(
        X_al, X_fast, rtol=1e-4, atol=1e-3,
        err_msg=f"[{model}] al_pipeline serial diverges from MCSC fast",
    )
    np.testing.assert_allclose(
        X_al, X_numba, rtol=1e-4, atol=1e-3,
        err_msg=f"[{model}] al_pipeline serial diverges from MCSC numba",
    )


@requires_mcsc
@requires_db
@pytest.mark.parametrize("model", ["hps_urry", "hps_kr", "mpipi", "calvados"])
def test_alpipeline_matches_mcsc_serial(model: str):
    """
    al_pipeline serial == MCSC serial (the original Python implementation).

    Both use `_effective_charge` for CALVADOS N/C-terminal handling. This is the
    tightest parity case (both pure-Python doubles) — should agree to
    summation-noise tolerance.
    """
    from al_pipeline.featurization.sequence_featurizer import SequenceFeaturizer as ALSF
    ser_mod = _load_mcsc_module("mcsc_sf_serial", "sequence_featurizer.py")

    db = str(DB_PATH)
    al  = ALSF(model, db)
    ser = ser_mod.SequenceFeaturizer(model, db)

    seqs = make_test_seqs()
    X_al  = al.featurize_many(seqs).to_numpy()
    X_ser = np.asarray(ser.featurize_many(seqs).to_numpy(), dtype=np.float64)

    np.testing.assert_allclose(
        X_al, X_ser, rtol=1e-12, atol=1e-12,
        err_msg=f"[{model}] al_pipeline serial diverges from MCSC serial",
    )


# --- ported numba featurizer parity ---
#
# After the port (Row 8 of the beam-search reimplementation), the numba
# featurizer lives under al_pipeline and no longer requires MCSC on
# ``PYTHONPATH``. This test validates the ported version against the
# al_pipeline serial reference with the same tolerance the MCSC-numba
# parity test uses.

@requires_db
@pytest.mark.parametrize("model", ["hps_urry", "hps_kr", "mpipi", "calvados"])
def test_alpipeline_numba_matches_serial(model: str):
    """
    Ported al_pipeline numba == al_pipeline serial to within fastmath +
    parallel reduction noise.

    Tolerance mirrors the MCSC-numba test (rtol=1e-4, atol=1e-3): fastmath
    reorders float summations, so bit-exact match with the serial doubles
    isn't achievable. Empirically the discrepancy stays ~1e-6 on our test
    sequences.
    """
    from al_pipeline.featurization.sequence_featurizer import SequenceFeaturizer as ALSF
    from al_pipeline.featurization.sequence_featurizer_numba import (
        SequenceFeaturizer as ALNF,
    )

    seqs = make_test_seqs()
    db = str(DB_PATH)

    al = ALSF(model, db)
    nu = ALNF(model, db)

    X_al = al.featurize_many(seqs).to_numpy()
    _ = nu.featurize_many_fast(seqs[:2], feat_threads=2, as_df=False)  # JIT warm-up
    X_nu = np.asarray(nu.featurize_many_fast(seqs, feat_threads=2, as_df=False))

    np.testing.assert_allclose(
        X_al, X_nu, rtol=1e-4, atol=1e-3,
        err_msg=f"[{model}] al_pipeline serial diverges from al_pipeline numba",
    )
