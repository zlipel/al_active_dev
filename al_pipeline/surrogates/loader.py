"""Public surrogate loader shared by AL (augmentation) and beam-search.

The AL augmentation path had a private `_load_surrogate_for_augment` in
`al_pipeline.ga.augmentation` that dispatched on ``cfg.train_model_type`` to
build the right `Surrogate`. Beam search needs the same dispatch — otherwise
its custom loader drifts from what AL trained against, which is the class of
bug the Row 8 cleanup exists to remove. Lifting the dispatch to a public
`load_surrogate` gives both callers a single code path.

The augmentation-side helper stays as a thin wrapper around this one during
the migration to preserve its return-signature contract (it also returns the
raw model_bundle / MoEBundle so the augmentation code can pass those into
retrain / save paths). Beam-search callers only need the `Surrogate` handle
and use it via `predict_pool` / `predict_design`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from al_pipeline.core.config import ALConfig
from al_pipeline.ga.ga_utils import (
    load_models,
    load_moe_bundle,
    load_normalization_stats,
)
from al_pipeline.surrogates.base import Surrogate
from al_pipeline.surrogates.gpr_global import make_surrogate

if TYPE_CHECKING:
    from al_pipeline.surrogates.moe import MoEBundle


def load_surrogate(
    cfg: ALConfig,
    *,
    temp: bool = False,
    device: str = "cpu",
) -> Surrogate:
    """Load the right `Surrogate` for ``cfg.train_model_type`` from disk.

    Parameters
    ----------
    cfg : ALConfig
        Consulted for ``train_model_type``, ``moe_policy``, ``moe_threshold``,
        ``obj1``, ``obj2``, and the checkpoint paths derived from
        ``cfg.paths``.
    temp : bool
        Load augmented checkpoints (True) rather than base ones (False).
        Beam search always uses base checkpoints; the augmentation code path
        toggles this depending on the kriging-believer stage.
    device : str
        Torch device for the loaded GP tensors. Default ``"cpu"`` covers the
        beam workload; augmentation callers may override.

    Returns
    -------
    Surrogate
        Live `MoESurrogate` or `GlobalGPRSurrogate` ready to serve
        `predict_pool` / `predict_design`.

    Notes
    -----
    Deliberately mirrors `augmentation._load_surrogate_for_augment` without
    exposing the underlying bundle to the caller — beam consumers only need
    the `Surrogate` handle. If a caller needs the bundle (as augmentation
    does for retraining), use `load_moe_bundle(cfg)` / `load_models(cfg,
    temp=temp)` directly.
    """
    t = cfg.train_model_type
    if t == "moe":
        moe_bundle: "MoEBundle" = load_moe_bundle(cfg, temp=temp)
        return make_surrogate(
            cfg,
            moe_bundle=moe_bundle,
            moe_policy=cfg.moe_policy,
            moe_threshold=cfg.moe_threshold,
        )
    # gpr_multitask / gpr_singletask share the same load path.
    normalization_stats = load_normalization_stats(cfg.paths.norm_stats)
    model_bundle = load_models(cfg, temp=temp, device=device)
    return make_surrogate(
        cfg,
        model_bundle=model_bundle,
        normalization_stats=normalization_stats,
    )
