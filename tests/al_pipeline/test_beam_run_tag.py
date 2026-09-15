"""run_tag naming consistency for the beam pipeline.

Both beam entrypoints (``prepare_endpoints.py`` and ``run_beams_mpi.py``) build
an ALPaths from CLI args. A set ``run_tag`` must suffix the AL artifacts
(features/labels/seq + MoE checkpoints) exactly like the AL pipeline, and an
empty ``run_tag`` must reproduce production names. Generic — no real files.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from al_pipeline.core.paths import ALPaths

_BEAM_DIR = Path(__file__).resolve().parents[2] / "beam_search"
if str(_BEAM_DIR) not in sys.path:
    sys.path.insert(0, str(_BEAM_DIR))

BASE_TAG = "epsilon_kriging_believer_yeoj_upper"


def _paths(run_tag, tmp):
    # Same field set both beam entrypoints construct.
    return ALPaths(
        base_path=tmp / "home",
        scratch_path=tmp / "scratch",
        iteration=10,
        front="upper",
        model="HPS_URRY",
        ehvi_variant="epsilon",
        exploration_strategy="kriging_believer",
        transform="yeoj",
        mc_ehvi=False,
        run_tag=run_tag,
    )


@pytest.mark.parametrize("run_tag", ["", "diag1", "soft_kb"])
def test_beam_artifacts_carry_run_tag(run_tag, tmp_path):
    p = _paths(run_tag, tmp_path)
    suffix = f"_{run_tag}" if run_tag else ""

    # Data artifacts (scratch side).
    assert p.features_csv.name == f"features_gen10{suffix}.csv"
    assert p.labels_csv.name == f"labels_gen10{suffix}.csv"
    assert p.seq_gen_txt.name == f"seq_gen10{suffix}.txt"

    # MoE checkpoints (home side) — the three files the loader pre-checks.
    assert p.moe_ps_chkpt(temp=False).name == f"MOE_PS_iter10_{BASE_TAG}{suffix}.pt"
    assert p.moe_nonps_chkpt(temp=False).name == f"MOE_NONPS_iter10_{BASE_TAG}{suffix}.pt"
    assert p.moe_rf_bundle(temp=False).name == f"MOE_RF_iter10_{BASE_TAG}{suffix}.pkl"


def test_loader_resolves_tagged_path(tmp_path):
    # The real beam loader consults the tagged path: with empty dirs it fails
    # the first existence check, naming the tagged features file.
    from cross_paths.model_io import load_beam_bundle

    p = _paths("diag1", tmp_path)
    with pytest.raises(FileNotFoundError, match=r"features_gen10_diag1\.csv"):
        load_beam_bundle(p, db_dir=tmp_path / "db")


def test_surrogate_cfg_carries_run_tag(tmp_path):
    # The ALConfig used to load the surrogate must resolve the same tagged MoE
    # checkpoints as the pre-check, or a tagged run loads untagged models.
    from cross_paths.model_io import _cfg_from_paths

    p = _paths("diag1", tmp_path)
    cfg = _cfg_from_paths(p, db_path=tmp_path / "db")
    assert cfg.run_tag == "diag1"
    assert cfg.paths.moe_ps_chkpt(temp=False) == p.moe_ps_chkpt(temp=False)
    assert cfg.paths.moe_rf_bundle(temp=False) == p.moe_rf_bundle(temp=False)
