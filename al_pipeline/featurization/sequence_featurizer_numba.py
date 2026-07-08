"""Batched sequence featurizer backed by Numba's parallel JIT.

Ported from MODEL_COMPARISON_STELLAR_CURR/sequence_featurizer_numba.py to
remove the ``PYTHONPATH`` dependency the beam search relied on. Algorithmic
behavior is preserved bit-for-bit against that source; parity vs. the pure-
Python :class:`al_pipeline.featurization.sequence_featurizer.SequenceFeaturizer`
holds within ``rtol=1e-4, atol=1e-3`` (fastmath + parallel reduction noise) —
see ``tests/al_pipeline/test_featurizer_parity.py``.

Public API mirrors the serial featurizer with one added batch method:

    fzr = SequenceFeaturizer(model_name, db_path)
    X = fzr.featurize_many_fast(sequences, feat_threads=N, as_df=False)

29-column output matches the serial layout: 20 AA counts +
[length, SCD, SHD, |net charge|, sum lambda, beads(+), beads(-), shan ent, mol wt].
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from importlib.machinery import SourceFileLoader
from typing import List, Tuple

import numba as nb

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_N = 20

MODEL_TO_FILE = {
    "hps_urry": "amino_acid_Urry.py",
    "hps_kr": "amino_acid_KR.py",
    "mpipi": "mpipi.py",
    "calvados": "calvados.py"
}

# ---------- fast ASCII->index lookup ----------
_AA_LUT = np.full(256, -1, dtype=np.int16)
for i, aa in enumerate(AMINO_ACIDS.encode("ascii")):
    _AA_LUT[aa] = i


def _pack_seqs(seqs: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pack list of strings into:
      buf: int32 flat array of AA indices
      offsets: int64 offsets (len = nseq+1)
      lengths: int32 lengths (len = nseq)
    """
    nseq = len(seqs)
    lengths = np.fromiter((len(s) for s in seqs), dtype=np.int32, count=nseq)
    offsets = np.empty(nseq + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    buf = np.empty(offsets[-1], dtype=np.int16)

    pos = 0
    for s in seqs:
        b = s.encode("ascii")
        arr = _AA_LUT[np.frombuffer(b, dtype=np.uint8)]
        # if you want safety:
        # if (arr < 0).any(): raise ValueError("Non-canonical AA found")
        buf[pos:pos + arr.size] = arr
        pos += arr.size

    return buf.astype(np.int32), offsets, lengths


@nb.njit(fastmath=True, nogil=True)
def _log2(x):
    return np.log(x) / np.log(2.0)

@nb.njit(inline='always')
def _effective_charge_at(buf, idx, start, end, charge, model_kind):
    aa = buf[idx]
    if model_kind == 2:  # calvados
        if idx == start:
            return charge[aa + AA_N]          # N-terminus
        elif idx == end - 1:
            return charge[aa + 2 * AA_N]      # C-terminus
    return charge[aa]


@nb.njit(parallel=True, fastmath=True, nogil=True)
def _featurize_batch(
    buf: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    charge: np.ndarray,
    lam_or_eps: np.ndarray,
    mass: np.ndarray,
    sqrt_d: np.ndarray,
    inv_d: np.ndarray,
    model_kind: int,           # 0=hps, 1=mpipi, 2=calvados
) -> np.ndarray:
    nseq = lengths.shape[0]
    out = np.empty((nseq, 29), dtype=np.float64)

    for k in nb.prange(nseq):
        start = offsets[k]
        end = offsets[k + 1]
        n = end - start
        invn = 1.0 / n

        counts = np.zeros(AA_N, dtype=np.int32)
        for t in range(start, end):
            counts[buf[t]] += 1

        for a in range(AA_N):
            out[k, a] = counts[a]

        out[k, 20] = n

        ent = 0.0
        for a in range(AA_N):
            c = counts[a]
            if c > 0:
                p = c * invn
                ent -= p * _log2(p)

        netq = 0.0
        pos = 0.0
        neg = 0.0
        mw = 0.0

        for t in range(start, end):
            aa = buf[t]
            q = _effective_charge_at(buf, t, start, end, charge, model_kind)
            netq += q
            if q > 0.0:
                pos += 1.0
            elif q < 0.0:
                neg += 1.0
            mw += mass[aa]

        abs_netq = abs(netq)

        scd = 0.0
        shd = 0.0
        sumlam = 0.0

        if model_kind == 1:  # mpipi
            for t in range(start, end):
                aa = buf[t]
                sumlam += lam_or_eps[aa * AA_N + aa]

            for i in range(n):
                ti = start + i
                ai = buf[ti]
                qi = _effective_charge_at(buf, ti, start, end, charge, model_kind)
                base_i = ai * AA_N

                for j in range(i + 1, n):
                    tj = start + j
                    d = j - i
                    aj = buf[tj]
                    qj = _effective_charge_at(buf, tj, start, end, charge, model_kind)
                    eps_ij = lam_or_eps[base_i + aj]
                    shd += eps_ij * inv_d[d]
                    scd += qi * qj * sqrt_d[d]

        else:  # hps + calvados
            for t in range(start, end):
                aa = buf[t]
                sumlam += lam_or_eps[aa]

            for i in range(n):
                ti = start + i
                ai = buf[ti]
                qi = _effective_charge_at(buf, ti, start, end, charge, model_kind)
                li = lam_or_eps[ai]

                for j in range(i + 1, n):
                    tj = start + j
                    d = j - i
                    aj = buf[tj]
                    qj = _effective_charge_at(buf, tj, start, end, charge, model_kind)
                    scd += qi * qj * sqrt_d[d]
                    shd += (li + lam_or_eps[aj]) * inv_d[d]

        out[k, 21] = scd * invn
        out[k, 22] = shd * invn
        out[k, 23] = abs_netq
        out[k, 24] = sumlam
        out[k, 25] = pos
        out[k, 26] = neg
        out[k, 27] = ent
        out[k, 28] = mw

    return out


class SequenceFeaturizer:
    def __init__(self, model_name: str, db_path: str):
        self.model = model_name
        self.db_path = db_path
        self._load_model_params()

    def _load_model_params(self):
        ff_db = SourceFileLoader('ff_db', f"{self.db_path}/ff_db.py").load_module()
        try:
            ff_db.import_parameters(
                f"{self.db_path}/{MODEL_TO_FILE[self.model]}", verbose=False,
            )
        except TypeError:
            ff_db.import_parameters(
                f"{self.db_path}/{MODEL_TO_FILE[self.model]}",
            )
        atm_types = ff_db.atm_types

        # model kind id for numba kernel
        if self.model == "mpipi":
            self._kind = 1
        elif self.model == "calvados":
            self._kind = 2
        else:
            self._kind = 0

        # charge + mass arrays (20)
        self.mass_array = np.zeros(AA_N, dtype=np.float64)
        self.charge_array = np.zeros(AA_N, dtype=np.float64)

        for i, aa in enumerate(AMINO_ACIDS):
            self.mass_array[i] = atm_types[aa]['m']
            self.charge_array[i] = atm_types[aa]['q']

        # calvados: need extra terminal charges to do the correction
        if self.model == "calvados":
            # layout: [aa], [aan], [aac]
            charge60 = np.zeros(AA_N * 3, dtype=np.float64)
            for i, aa in enumerate(AMINO_ACIDS):
                charge60[i] = atm_types[aa]['q']
                charge60[i + AA_N] = atm_types[f"{aa}n"]['q']
                charge60[i + 2 * AA_N] = atm_types[f"{aa}c"]['q']
            self.charge_array = charge60  # overwrite with 60-len for kernel

        # lam / eps
        if self.model == "mpipi":
            nonbon = ff_db.nonbon_types
            eps = np.zeros(AA_N * AA_N, dtype=np.float64)
            for i, aa in enumerate(AMINO_ACIDS):
                for j, bb in enumerate(AMINO_ACIDS):
                    pair = tuple(sorted((aa, bb)))
                    if pair in nonbon:
                        eps[i * AA_N + j] = nonbon[pair]['eps'] / 0.2
            self.lam_or_eps = eps
        else:
            lam = np.zeros(AA_N, dtype=np.float64)
            for i, aa in enumerate(AMINO_ACIDS):
                lam[i] = atm_types[aa]['lam']
            self.lam_or_eps = lam

        del ff_db

    def featurize_many_fast(self, sequences: List[str], feat_threads: int = 1, as_df: bool = False):
        """
        Fast batched featurization via Numba prange across sequences.

        Parameters
        ----------
        sequences : list[str]
        feat_threads : int
            Number of threads Numba will use inside this *process*.
            IMPORTANT: for MPI, this should match cpus-per-task and you must avoid oversubscription.
        as_df : bool
            If True, returns a DataFrame; else returns np.ndarray (N,29).

        Returns
        -------
        np.ndarray or pd.DataFrame
        """
        if feat_threads is None or feat_threads <= 0:
            feat_threads = 1

        # pack sequences to contiguous arrays
        buf, offsets, lengths = _pack_seqs(sequences)

        # precompute distance weights up to max length
        Lmax = int(lengths.max()) if len(lengths) else 0
        sqrt_d = np.zeros(Lmax + 1, dtype=np.float32)
        inv_d  = np.zeros(Lmax + 1, dtype=np.float32)
        for d in range(1, Lmax + 1):
            sqrt_d[d] = np.sqrt(d)
            inv_d[d]  = 1.0 / d

        X = _featurize_batch(
            buf, offsets, lengths,
            self.charge_array.astype(np.float32),
            self.lam_or_eps.astype(np.float32),
            self.mass_array.astype(np.float32),
            sqrt_d, inv_d,
            int(self._kind),
        )

        if not as_df:
            return X

        columns = [aa for aa in AMINO_ACIDS] + [
            "length", "SCD", "SHD", "|net charge|", "sum lambda",
            "beads(+)", "beads(-)", "shan ent", "mol wt"
        ]
        return pd.DataFrame(X, columns=columns)
