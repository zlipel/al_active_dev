"""Gate 1 — Row 8 inference plumbing self-consistency check.

Per §IV.acceptance of the beam-search plan: the beam's `predict_design`
output for a labeled sequence must equal a direct `GPRExpert` inference
(bypassing the `Surrogate` ABC) on the same sequence, to numerical
precision. If they differ, Row 8's plumbing (persisted-scaler load,
featurizer port, feature-normalizer pass-through) has drifted from AL's
live inference path.

Runs against the live MoE bundles under
``runs/<MODEL>/MODELS/MOE_{PS,NONPS,RF}_iter10_epsilon_kriging_believer_yeoj_upper.*``
plus the labeled features/labels CSVs. Skips automatically if any
required artifact is missing.

Cluster-marked because the artifact tree is large and env-specific; run
with ``CLUSTER_TESTS=1 pytest tests/al_pipeline/test_gate1_selfconsistency.py``
locally (after rsyncing bundles + labels) or on Stellar directly.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from al_pipeline.core.config import ALConfig
from al_pipeline.core.paths import ALPaths
from al_pipeline.surrogates import (
    GPRExpert,
    MoESurrogate,
    load_surrogate,
)


MODELS = ("CALVADOS", "HPS_URRY", "MPIPI")
DEFAULT_ITER = 10
DEFAULT_FRONT = "upper"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _paths_for(model: str, iteration: int = DEFAULT_ITER) -> ALPaths:
    """Resolve the beam bundle + labels paths.

    On the cluster, ``HOME_AL`` and ``SCRATCH_AL`` come from ``config/cluster.env``
    and point at the runs / scratch trees. Locally, the labeled CSVs live
    on cluster scratch by default — set ``SCRATCH_AL`` to a local mirror if
    you've rsync'd them, otherwise the test skips.
    """
    root = _repo_root()
    home_al = Path(os.environ.get("HOME_AL", root / "runs"))
    scratch_al = Path(os.environ.get("SCRATCH_AL", root / "runs"))
    return ALPaths(
        base_path=home_al,
        scratch_path=scratch_al,
        iteration=iteration,
        front=DEFAULT_FRONT,
        model=model,
        ehvi_variant="epsilon",
        exploration_strategy="kriging_believer",
        transform="yeoj",
        mc_ehvi=False,
    )


def _cfg_for(paths: ALPaths, db_path: Path) -> ALConfig:
    return ALConfig(
        model=paths.model,
        iteration=paths.iteration,
        front=paths.front,
        ehvi_variant=paths.ehvi_variant,
        exploration_strategy=paths.exploration_strategy,
        transform=paths.transform,
        mc_ehvi=paths.mc_ehvi,
        base_path=Path(paths.base_path),
        scratch_path=Path(paths.scratch_path),
        db_path=db_path,
        train_model_type="moe",
        obj1="exp_density",
        obj2="diff",
        aux1_obj1="density",
    )


def _find_db_path() -> Path | None:
    env = os.environ.get("DB_PATH")
    if env and (Path(env) / "ff_db.py").exists():
        return Path(env)
    for candidate in (
        Path("/Users/zl4808/Documents/ActiveLearningAndIDPs/stellar_scripts/GENDATA/databases"),
        Path("/home/zl4808/scripts/GENDATA/databases"),
    ):
        if (candidate / "ff_db.py").exists():
            return candidate
    return None


def _artifacts_present(paths: ALPaths) -> bool:
    return all(
        Path(p).exists()
        for p in (
            paths.moe_ps_chkpt(temp=False),
            paths.moe_nonps_chkpt(temp=False),
            paths.moe_rf_bundle(temp=False),
            paths.features_csv,
            paths.labels_csv,
        )
    )


@pytest.mark.cluster
@pytest.mark.parametrize("model", MODELS)
def test_gate1_predict_design_matches_direct_expert(model: str):
    """`surrogate.predict_design(X_labeled).per_expert[r].phys_mean` equals
    a direct `GPRExpert.from_checkpoint(<expert.pt>)` prediction inverse-
    transformed via its persisted scalers, to numerical precision.

    Passing this gate proves:
      * persisted scalers loaded, not refit (Row 8 fix #3)
      * feature normalizer pass-through inside `predict_design` matches the
        expert's own `_feature_tensor` (Row 8 fix #2)
      * numba featurizer produces the same z as the training-time
        featurizer on the labeled rows (Row 8 fix #1)
    """
    db_path = _find_db_path()
    if db_path is None:
        pytest.skip("gendata FF database not found (set DB_PATH env var)")

    paths = _paths_for(model)
    if not _artifacts_present(paths):
        pytest.skip(
            f"[{model}] MoE bundle or labels CSV missing under {paths.base_path}"
        )

    cfg = _cfg_for(paths, db_path)

    # --- surrogate side ---
    surrogate = load_surrogate(cfg, temp=False)
    assert isinstance(surrogate, MoESurrogate), (
        f"[{model}] load_surrogate returned {type(surrogate).__name__}, expected MoESurrogate"
    )

    features_df = pd.read_csv(paths.features_csv)
    labels_df = pd.read_csv(paths.labels_csv)
    # Drop rows with any NaN in the objectives + regime column (matches
    # moe_training's clean-then-save contract).
    labels_clean = labels_df.dropna(subset=["exp_density", "diff", "density"])
    features_clean = features_df.loc[labels_clean.index].reset_index(drop=True)
    labels_clean = labels_clean.reset_index(drop=True)

    pred = surrogate.predict_design(features_clean)

    # --- direct expert side ---
    ps_ckpt = torch.load(
        paths.moe_ps_chkpt(temp=False), map_location="cpu", weights_only=False,
    )
    nps_ckpt = torch.load(
        paths.moe_nonps_chkpt(temp=False), map_location="cpu", weights_only=False,
    )
    direct_ps = GPRExpert.from_checkpoint(
        ps_ckpt, str(paths.features_csv), str(paths.labels_csv),
    )
    direct_nps = GPRExpert.from_checkpoint(
        nps_ckpt, str(paths.features_csv), str(paths.labels_csv),
    )

    for regime, direct in (("ps", direct_ps), ("nonps", direct_nps)):
        direct_out = direct.predict(features_clean)
        direct_z_mean = np.column_stack([
            direct_out["exp_density_z_mean"],
            direct_out["diff_z_mean"],
        ])
        direct_phys = direct.inverse_scale_z(direct_z_mean)

        surrogate_z = pred.per_expert[regime]["z_mean"]
        surrogate_phys = pred.per_expert[regime]["phys_mean"]

        np.testing.assert_allclose(
            surrogate_z, direct_z_mean, rtol=1e-6, atol=1e-8,
            err_msg=f"[{model}/{regime}] z_mean drift between surrogate and direct expert",
        )
        np.testing.assert_allclose(
            surrogate_phys, direct_phys, rtol=1e-6, atol=1e-8,
            err_msg=f"[{model}/{regime}] phys_mean drift between surrogate and direct expert",
        )


@pytest.mark.cluster
@pytest.mark.parametrize("model", MODELS)
def test_gate1_oof_smell_beam_at_least_as_accurate_as_kfold(model: str):
    """Full-data expert (what the beam uses) should predict labeled PS
    sequences at least as accurately as the OOF (k-fold, held-out expert).

    Not a hard gate — a "corruption detector" after the strict Gate 1
    passes. If the full-data expert's mean absolute error on labeled PS
    sequences exceeds the OOF's error, something in Row 8's plumbing is
    corrupted even though the self-consistency check passes (e.g. wrong
    features_csv version used, silently truncated labels).
    """
    db_path = _find_db_path()
    if db_path is None:
        pytest.skip("gendata FF database not found (set DB_PATH env var)")

    paths = _paths_for(model)
    if not _artifacts_present(paths):
        pytest.skip(
            f"[{model}] MoE bundle or labels CSV missing under {paths.base_path}"
        )

    oof_csv = (
        Path(paths.base_path)
        / model / "DIAGNOSTIC" / f"regime_oof_predictions_iter{paths.iteration}.csv"
    )
    if not oof_csv.exists():
        pytest.skip(f"[{model}] OOF predictions CSV missing at {oof_csv}")

    cfg = _cfg_for(paths, db_path)
    surrogate = load_surrogate(cfg, temp=False)

    features_df = pd.read_csv(paths.features_csv)
    labels_df = pd.read_csv(paths.labels_csv)
    labels_clean = labels_df.dropna(subset=["exp_density", "diff", "density"])
    features_clean = features_df.loc[labels_clean.index].reset_index(drop=True)
    labels_clean = labels_clean.reset_index(drop=True)

    pred = surrogate.predict_design(features_clean)

    # Restrict to strong-PS rows (true_is_ps == 1) — OOF's PS expert column
    # is only meaningful for those (nonPS true labels get predicted by a
    # PS-expert that never saw a matching label distribution).
    oof = pd.read_csv(oof_csv)
    ps_rows = oof[oof["true_is_ps"] == 1].reset_index(drop=True)
    if len(ps_rows) < 20:
        pytest.skip(f"[{model}] fewer than 20 PS OOF rows; smell test not meaningful")

    # Align: OOF has `original_index` back into labels_clean rows.
    idx = ps_rows["original_index"].to_numpy().astype(int)
    true_rho = labels_clean.loc[idx, "exp_density"].to_numpy()
    beam_rho_phys = pred.per_expert["ps"]["phys_mean"][idx, 0]
    oof_rho_phys = ps_rows["pred_exp_density_ps_expert"].to_numpy()

    mae_beam = float(np.mean(np.abs(beam_rho_phys - true_rho)))
    mae_oof = float(np.mean(np.abs(oof_rho_phys - true_rho)))
    print(
        f"\n  [{model}] labeled-PS MAE (exp_density): "
        f"beam full-data expert = {mae_beam:.4f}, OOF k-fold = {mae_oof:.4f}"
    )
    assert mae_beam <= mae_oof + 1e-3, (
        f"[{model}] beam full-data expert MAE ({mae_beam:.4f}) exceeds OOF "
        f"MAE ({mae_oof:.4f}) on labeled PS sequences — Row 8 plumbing "
        "may be feeding the surrogate stale features or labels."
    )
