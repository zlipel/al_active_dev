"""Beam-search policy layer — Row 9 of the beam-search reimplementation.

One `BeamPolicy` object wraps the substrate laid by Row 8 (surrogate,
featurizer, quantile transforms) and exposes a single `predict_candidates`
method that the beam loop calls at every step. Five kinds are reachable
through the same dispatch:

  * ``expert_tied``  — start regime picks one expert; scores every
    candidate through that expert for the whole walk. Records endpoint
    ``p_ps`` as a drift diagnostic but never rejects on it. **Primary policy
    validated in this branch** (§III.1).
  * ``anchored_reject`` — same expert selection as ``expert_tied`` but
    rejects candidates the classifier confidently says are opposite-regime
    (``p_ps`` below/above ``reject_threshold``). Reachable via ``--policy``
    but not validated here.
  * ``soft`` — probabilistic mixture per ``MoESurrogate`` (``predict_design``
    blended mean). Reachable but not validated.
  * ``hard`` — deterministic expert switch per candidate at
    ``hard_threshold``. Reachable but not validated.
  * ``global`` — single global GPR baseline. Reachable but no physical
    inversion path exists yet (`GlobalGPRSurrogate.predict_design` returns
    ``phys_mean=None``); every candidate ends up ``invalid_phys``. Row 8
    documents the follow-up.

The policy owns the featurizer + `q_rho` / `q_diff` transforms so
`predict_candidates` is the one-and-only per-step call site. `beam_search_paths`
holds a reference to the policy and no longer reaches into a raw bundle for
scalers — the policy exposes ``label_scalers`` as a convenience for computing
the start's z-space coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

from al_pipeline.surrogates import Surrogate
from al_pipeline.surrogates.gpr_global import GlobalGPRSurrogate
from al_pipeline.surrogates.moe import MoESurrogate


Kind = Literal["expert_tied", "anchored_reject", "soft", "hard", "global"]

REGIMES = ("ps", "nonps")


@dataclass(frozen=True)
class PolicyPrediction:
    """Per-step batch output consumed by the beam loop.

    Shape convention matches Row 8's `predict_design`: ``(B, 2)`` for
    per-objective arrays, ``(B,)`` for scalars.

    Attributes
    ----------
    z_mean, z_std
        Normalized objective space per candidate. The policy either takes
        these from the blended surrogate output (soft), the selected
        expert (expert_tied / anchored_reject), or a per-candidate switch
        (hard).
    phys
        Inverse-scaled physical means, ``(B, 2)``. ``NaN`` where invalid.
    uv
        Quantile-transformed physical means, ``(B, 2)``. ``NaN`` where
        invalid (i.e. where ``phys`` is ``NaN`` — the ``QuantileTransformer``
        is only applied to rows where ``valid[i] == True``).
    p_ps
        Gate probability, ``(B,)``. ``None`` when the surrogate has no gate
        (i.e. ``global``).
    valid
        Boolean mask, ``(B,)``. ``True`` means the candidate passes both
        physical-validity and (for ``anchored_reject``) the gate filter.
    reason
        Per-candidate string reason code (object dtype), ``(B,)``.
        Values: ``"ok"``, ``"invalid_phys"``, ``"rejected_by_gate"``.
    """
    z_mean: np.ndarray
    z_std: np.ndarray
    phys: np.ndarray
    uv: np.ndarray
    p_ps: np.ndarray | None
    valid: np.ndarray
    reason: np.ndarray


class BeamPolicy:
    """Beam-search prediction policy over one `Surrogate`."""

    def __init__(
        self,
        *,
        kind: Kind,
        surrogate: Surrogate,
        featurizer: Any,
        q_rho: Any,
        q_diff: Any,
        start_regime: str | None = None,
        hard_threshold: float = 0.5,
        reject_threshold: float = 0.5,
        min_positive: float = 1e-12,
        feat_threads: int = 1,
    ):
        if kind not in ("expert_tied", "anchored_reject", "soft", "hard", "global"):
            raise ValueError(f"unknown BeamPolicy kind={kind!r}")
        if kind in ("expert_tied", "anchored_reject") and start_regime not in REGIMES:
            raise ValueError(
                f"kind={kind!r} requires start_regime in {REGIMES}; got {start_regime!r}"
            )
        if kind in ("soft", "hard", "anchored_reject", "expert_tied") and not isinstance(surrogate, MoESurrogate):
            raise TypeError(
                f"kind={kind!r} requires an MoESurrogate (needs per-expert access + gate); "
                f"got {type(surrogate).__name__}"
            )
        if kind == "global" and not isinstance(surrogate, GlobalGPRSurrogate):
            raise TypeError(
                f"kind='global' requires a GlobalGPRSurrogate; got {type(surrogate).__name__}"
            )
        self.kind: Kind = kind
        self.surrogate = surrogate
        self.featurizer = featurizer
        self.q_rho = q_rho
        self.q_diff = q_diff
        self.start_regime = start_regime
        self.hard_threshold = float(hard_threshold)
        self.reject_threshold = float(reject_threshold)
        self.min_positive = float(min_positive)
        self.feat_threads = int(feat_threads)

    # ------------------------------------------------------------------
    # Convenience accessors (kept to keep beam_search_paths bundle-free)
    # ------------------------------------------------------------------

    @property
    def label_scalers(self) -> tuple:
        """Shared ``(label_scaler1, label_scaler2)`` for physical <-> z conversion.

        Both experts hold identical instances under
        ``label_scaler_scope='all'`` (Row 8 substrate), so reading them off
        the PS expert is safe. `global` surrogates don't persist scalers and
        raise here — `beam_search_paths` only queries this to place the
        *start* sequence in z-space, and a global beam run without persisted
        scalers can't do that anyway.
        """
        if isinstance(self.surrogate, MoESurrogate):
            ex = self.surrogate.bundle.ps_expert
            return (ex.label_scaler1, ex.label_scaler2)
        raise RuntimeError(
            "GlobalGPRSurrogate does not persist label scalers; start-z0 "
            "cannot be inferred from physical. Provide start_z explicitly."
        )

    # ------------------------------------------------------------------
    # Per-step prediction
    # ------------------------------------------------------------------

    def predict_candidates(self, seqs: Iterable[str]) -> PolicyPrediction:
        """Featurize + score a batch of candidate sequences under the policy."""
        seqs = list(seqs)
        B = len(seqs)
        if B == 0:
            empty2 = np.zeros((0, 2), dtype=np.float64)
            return PolicyPrediction(
                z_mean=empty2, z_std=empty2, phys=empty2, uv=empty2,
                p_ps=None, valid=np.zeros(0, dtype=bool),
                reason=np.zeros(0, dtype=object),
            )

        feat_threads_eff = 1 if B < 64 else self.feat_threads
        X: pd.DataFrame = self.featurizer.featurize_many_fast(
            seqs, feat_threads_eff, as_df=True,
        )
        pred = self.surrogate.predict_design(X)

        z_mean, z_std, phys = self._select_channel(pred)
        p_ps = pred.p_ps

        # Physical validity: finite + strictly positive on both axes.
        reason = np.full(B, "ok", dtype=object)
        if phys is None:
            phys = np.full((B, 2), np.nan, dtype=np.float64)
            valid = np.zeros(B, dtype=bool)
            reason[:] = "invalid_phys"
        else:
            finite = np.isfinite(phys[:, 0]) & np.isfinite(phys[:, 1])
            positive = (phys[:, 0] > self.min_positive) & (phys[:, 1] > self.min_positive)
            valid = finite & positive
            reason[~valid] = "invalid_phys"

        # Anchored reject: filter candidates the gate confidently attributes
        # to the opposite regime. Only applied on top of a physical-valid
        # candidate; invalid_phys takes precedence.
        if self.kind == "anchored_reject" and p_ps is not None:
            if self.start_regime == "ps":
                gate_reject = p_ps < self.reject_threshold
            else:  # nonps
                gate_reject = p_ps >= self.reject_threshold
            newly_rejected = gate_reject & valid
            valid = valid & ~newly_rejected
            reason[newly_rejected] = "rejected_by_gate"

        # Quantile transform on the physical means, per-row, only for valid rows.
        uv = np.full((B, 2), np.nan, dtype=np.float64)
        if np.any(valid):
            uv[valid, 0] = self.q_rho.transform(phys[valid, 0].reshape(-1, 1)).ravel()
            uv[valid, 1] = self.q_diff.transform(phys[valid, 1].reshape(-1, 1)).ravel()

        return PolicyPrediction(
            z_mean=np.asarray(z_mean, dtype=np.float64),
            z_std=np.asarray(z_std, dtype=np.float64),
            phys=phys,
            uv=uv,
            p_ps=(np.asarray(p_ps, dtype=np.float64) if p_ps is not None else None),
            valid=valid,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Kind-specific channel extraction from a DesignPrediction
    # ------------------------------------------------------------------

    def _select_channel(self, pred) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Return ``(z_mean, z_std, phys)`` per policy kind."""
        if self.kind in ("expert_tied", "anchored_reject"):
            per = pred.per_expert[self.start_regime]
            return per["z_mean"], per["z_std"], per["phys_mean"]

        if self.kind == "soft":
            return pred.z_mean, pred.z_std, pred.phys_mean

        if self.kind == "hard":
            # Per-candidate expert switch at self.hard_threshold. Independent
            # of the surrogate's own hard-policy configuration; the beam sets
            # the threshold explicitly so the surrogate policy stays a
            # separable knob (AL uses soft; beam picks its own switch).
            if pred.p_ps is None:
                raise RuntimeError("hard policy requires a gated surrogate (p_ps was None)")
            use_ps = (pred.p_ps >= self.hard_threshold)[:, None]
            per_ps = pred.per_expert["ps"]
            per_nonps = pred.per_expert["nonps"]
            z_mean = np.where(use_ps, per_ps["z_mean"], per_nonps["z_mean"])
            z_std = np.where(use_ps, per_ps["z_std"], per_nonps["z_std"])
            phys = np.where(use_ps, per_ps["phys_mean"], per_nonps["phys_mean"])
            return z_mean, z_std, phys

        if self.kind == "global":
            # phys_mean is None on the global surrogate until scaler persistence
            # lands (Row 8 documented). The policy still returns z-space for
            # ranking-agnostic consumers, but valid=False for all candidates.
            return pred.z_mean, pred.z_std, pred.phys_mean

        raise AssertionError(f"unhandled kind={self.kind!r}")  # pragma: no cover
