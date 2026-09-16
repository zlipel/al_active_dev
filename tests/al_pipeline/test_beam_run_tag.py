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


def _write(path, text="x\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_beam_reads_tagged_data_when_present(tmp_path):
    # With the tagged data files on disk, the resolved read paths pick them up.
    p = _paths("diag1", tmp_path)
    d = p.iter_scratch_dir
    _write(d / "features_gen10_diag1.csv")
    _write(d / "labels_gen10_diag1.csv")
    _write(d / "seq_gen10_diag1.txt")
    assert p.features_csv_resolved.name == "features_gen10_diag1.csv"
    assert p.labels_csv_resolved.name == "labels_gen10_diag1.csv"
    assert p.seq_gen_txt_resolved.name == "seq_gen10_diag1.txt"


def test_beam_falls_back_to_untagged_data_with_tagged_models(tmp_path):
    # A tagged run with only UNtagged data files: the read paths fall back to
    # the untagged data (writer paths stay tagged), and the loader gets past the
    # data pre-check to fail on the missing TAGGED model — so untagged data is
    # accepted while the tagged model variant is still required.
    from cross_paths.model_io import load_beam_bundle

    p = _paths("moe_soft", tmp_path)
    d = p.iter_scratch_dir
    _write(d / "features_gen10.csv")
    _write(d / "labels_gen10.csv")
    _write(d / "seq_gen10.txt")

    assert p.features_csv_resolved.name == "features_gen10.csv"     # read: fallback
    assert p.features_csv.name == "features_gen10_moe_soft.csv"     # write: tagged

    with pytest.raises(FileNotFoundError, match=r"MOE_PS_iter10_.*moe_soft\.pt"):
        load_beam_bundle(p, db_dir=tmp_path / "db")


def test_beam_missing_data_reports_untagged_name(tmp_path):
    # Neither tagged nor untagged present -> resolved returns the untagged base.
    from cross_paths.model_io import load_beam_bundle

    p = _paths("diag1", tmp_path)
    with pytest.raises(FileNotFoundError, match=r"features_gen10\.csv"):
        load_beam_bundle(p, db_dir=tmp_path / "db")


def test_loader_uses_raw_features_not_norm(tmp_path):
    # The beam loads the raw features/labels, never the _NORM_ training variants.
    from cross_paths.model_io import load_beam_bundle

    p = _paths("", tmp_path)
    assert p.features_csv != p.features_norm_csv
    assert p.labels_csv != p.labels_norm_csv
    assert "_NORM_" not in p.features_csv.name
    assert "_NORM_" not in p.labels_csv.name

    # With only the NORM file present the loader still requires the raw file
    # (no NORM fallback): it fails naming the raw features CSV.
    p.features_norm_csv.parent.mkdir(parents=True, exist_ok=True)
    p.features_norm_csv.write_text("stub\n")
    with pytest.raises(FileNotFoundError, match=r"features_gen10\.csv"):
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
