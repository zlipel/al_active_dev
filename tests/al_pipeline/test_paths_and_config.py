"""
Boundary contract tests for ALConfig / ALPaths.

These pin the names that downstream simulation/analysis scripts read against:
  - candidate file name pattern: simulation_candidates_gen{N}_{front}.txt
  - csv names: eos_results.csv, diffusivities.csv
  - default obj1 / obj2: exp_density and diff (must match the CSV columns produced
    by process_eos_sims.py / process_diff_sims.py)

If any of these drift, the simulation submit scripts will silently look in the
wrong place.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from al_pipeline.core.paths import ALPaths, _tag
from al_pipeline.core.config import ALConfig


def _paths(tmp_path: Path, *, iteration: int = 0, front: str = "upper") -> ALPaths:
    return ALPaths(
        base_path=tmp_path / "home",
        scratch_path=tmp_path / "scratch",
        iteration=iteration,
        front=front,
        model="hps_urry",
    )


# ---------- ALPaths name contracts ----------

def test_eos_csv_name_is_stable(tmp_path):
    """eos_results.csv is the literal name process_eos_sims.py writes — must not drift."""
    p = _paths(tmp_path)
    assert p.eos_csv.name == "eos_results.csv"


def test_diff_csv_name_is_stable(tmp_path):
    """diffusivities.csv is the literal name process_diff_sims.py writes."""
    p = _paths(tmp_path)
    assert p.diff_csv.name == "diffusivities.csv"


def test_next_iter_candidates_filename_pattern(tmp_path):
    """make_eos.sh reads `simulation_candidates_gen{N}_{front}.txt`. Lock the format."""
    for it in (0, 1, 5):
        for front in ("upper", "lower"):
            p = _paths(tmp_path, iteration=it, front=front)
            expected = f"simulation_candidates_gen{it + 1}_{front}.txt"
            assert p.next_iter_candidates_file.name == expected


def test_next_iter_dirs_under_simulations_subtree(tmp_path):
    """The candidate file mirrors must land under SIMULATIONS/{EOS,DIFF}/."""
    p = _paths(tmp_path, iteration=2, front="upper")
    assert p.next_iter_eos_dir.parts[-2:]  == ("SIMULATIONS", "EOS")
    assert p.next_iter_diff_dir.parts[-2:] == ("SIMULATIONS", "DIFF")


# ---------- tag construction ----------

def test_tag_excludes_mc_suffix_when_analytic():
    tag = _tag("epsilon", "kriging_believer", "yeoj", "upper", mc_ehvi=False)
    assert tag == "epsilon_kriging_believer_yeoj_upper"


def test_tag_includes_mc_suffix_when_mc():
    tag = _tag("epsilon", "kriging_believer", "yeoj", "upper", mc_ehvi=True)
    assert tag == "epsilon_kriging_believer_yeoj_upper_mc"


def test_alpaths_tag_property_matches_tag_function(tmp_path):
    p = _paths(tmp_path)
    assert p.tag == _tag(p.ehvi_variant, p.exploration_strategy, p.transform, p.front, p.mc_ehvi)


# ---------- ALConfig objective contract ----------

def test_default_obj1_obj2_match_analysis_csv_columns():
    """obj1='exp_density' and obj2='diff' must match the column names used by
    process_eos_sims.py / process_diff_sims.py to populate their CSVs."""
    cfg = ALConfig(model="hps_urry", iteration=0, front="upper")
    assert cfg.obj1 == "exp_density"
    assert cfg.obj2 == "diff"


def test_default_aux1_obj1_is_density():
    """aux1_obj1='density' is the auxiliary column read alongside exp_density."""
    cfg = ALConfig(model="hps_urry", iteration=0, front="upper")
    assert cfg.aux1_obj1 == "density"


def test_validate_rejects_identical_objectives():
    cfg = ALConfig(model="hps_urry", iteration=0, front="upper", obj1="x", obj2="x")
    with pytest.raises(ValueError, match="different"):
        cfg.validate()


def test_validate_rejects_negative_iteration():
    cfg = ALConfig(model="hps_urry", iteration=-1, front="upper")
    with pytest.raises(ValueError, match="non-negative"):
        cfg.validate()


def test_paths_property_returns_alpaths_with_matching_fields():
    cfg = ALConfig(model="hps_urry", iteration=3, front="lower")
    p = cfg.paths
    assert p.iteration == 3
    assert p.front == "lower"
    assert p.model == "hps_urry"
