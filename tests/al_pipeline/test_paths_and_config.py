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


# ---------- run_tag isolation (forward-convergence test) ----------

def test_run_tag_empty_reproduces_bare_tag(tmp_path):
    """Default run_tag='' must leave tag (and thus every _{tag} path) unchanged."""
    p = _paths(tmp_path)
    assert p.run_tag == ""
    assert p.tag == "epsilon_kriging_believer_yeoj_upper"


def test_run_tag_appends_to_tag(tmp_path):
    p = ALPaths(base_path=tmp_path / "home", scratch_path=tmp_path / "scratch",
                iteration=10, front="upper", model="hps_urry", run_tag="soft")
    assert p.tag == "epsilon_kriging_believer_yeoj_upper_soft"


def test_run_tag_threads_into_front_only_paths(tmp_path):
    """The two paths not keyed by tag (candidates file, logs dir) must also carry
    run_tag so policies don't collide there; empty run_tag matches production."""
    bare = _paths(tmp_path, iteration=10, front="upper")
    tagged = ALPaths(base_path=tmp_path / "home", scratch_path=tmp_path / "scratch",
                     iteration=10, front="upper", model="hps_urry", run_tag="hard")
    assert bare.next_iter_candidates_file.name == "simulation_candidates_gen11_upper.txt"
    assert tagged.next_iter_candidates_file.name == "simulation_candidates_gen11_upper_hard.txt"
    assert bare.logs_dir.name == "iteration_upper_10"
    assert tagged.logs_dir.name == "iteration_upper_10_hard"


def test_config_threads_run_tag_into_paths():
    cfg = ALConfig(model="hps_urry", iteration=10, front="upper", run_tag="soft")
    assert cfg.paths.run_tag == "soft"
    assert cfg.paths.tag.endswith("_soft")


# ---------- run_tag on cumulative generation artifacts + read fallback ----------

def _tagged_paths(tmp_path: Path, run_tag: str, iteration: int = 11) -> ALPaths:
    return ALPaths(base_path=tmp_path / "home", scratch_path=tmp_path / "scratch",
                   iteration=iteration, front="upper", model="hps_urry", run_tag=run_tag)


def test_cumulative_artifacts_untagged_when_run_tag_empty(tmp_path):
    p = _tagged_paths(tmp_path, "", iteration=11)
    assert p.features_csv.name == "features_gen11.csv"
    assert p.labels_csv.name == "labels_gen11.csv"
    assert p.seq_gen_txt.name == "seq_gen11.txt"
    assert p.eos_csv.name == "eos_results.csv"
    assert p.diff_csv.name == "diffusivities.csv"


def test_cumulative_outputs_tagged_when_run_tag_set(tmp_path):
    p = _tagged_paths(tmp_path, "moe_soft", iteration=11)
    assert p.features_csv.name == "features_gen11_moe_soft.csv"
    assert p.labels_csv.name == "labels_gen11_moe_soft.csv"
    assert p.seq_gen_txt.name == "seq_gen11_moe_soft.txt"


def test_tagged_reads_fall_back_to_untagged(tmp_path):
    """With run_tag set but only un-tagged files on disk, reads resolve un-tagged."""
    p = _tagged_paths(tmp_path, "moe_soft", iteration=11)
    p.eos_dir.mkdir(parents=True, exist_ok=True)
    p.diff_dir.mkdir(parents=True, exist_ok=True)
    p.prev_iter_scratch_dir.mkdir(parents=True, exist_ok=True)
    for f in (p.eos_dir / "eos_results.csv", p.diff_dir / "diffusivities.csv",
              p.eos_dir / "seq_gen11.txt",
              p.prev_iter_scratch_dir / "features_gen10.csv",
              p.prev_iter_scratch_dir / "labels_gen10.csv",
              p.prev_iter_scratch_dir / "seq_gen10.txt"):
        f.write_text("x")
    assert p.eos_csv.name == "eos_results.csv"
    assert p.diff_csv.name == "diffusivities.csv"
    assert p.eos_seq_gen_txt.name == "seq_gen11.txt"
    assert p.prev_features_csv.name == "features_gen10.csv"
    assert p.prev_labels_csv.name == "labels_gen10.csv"
    assert p.prev_seq_gen_txt.name == "seq_gen10.txt"


def test_tagged_reads_prefer_tagged_when_present(tmp_path):
    """When the tagged file exists it wins over the un-tagged default."""
    p = _tagged_paths(tmp_path, "moe_soft", iteration=11)
    p.eos_dir.mkdir(parents=True, exist_ok=True)
    p.diff_dir.mkdir(parents=True, exist_ok=True)
    p.prev_iter_scratch_dir.mkdir(parents=True, exist_ok=True)
    for name, d in [("eos_results.csv", p.eos_dir), ("eos_results_moe_soft.csv", p.eos_dir),
                    ("diffusivities.csv", p.diff_dir), ("diffusivities_moe_soft.csv", p.diff_dir),
                    ("seq_gen11.txt", p.eos_dir), ("seq_gen11_moe_soft.txt", p.eos_dir),
                    ("features_gen10.csv", p.prev_iter_scratch_dir),
                    ("features_gen10_moe_soft.csv", p.prev_iter_scratch_dir),
                    ("labels_gen10.csv", p.prev_iter_scratch_dir),
                    ("labels_gen10_moe_soft.csv", p.prev_iter_scratch_dir),
                    ("seq_gen10.txt", p.prev_iter_scratch_dir),
                    ("seq_gen10_moe_soft.txt", p.prev_iter_scratch_dir)]:
        (d / name).write_text("x")
    assert p.eos_csv.name == "eos_results_moe_soft.csv"
    assert p.diff_csv.name == "diffusivities_moe_soft.csv"
    assert p.eos_seq_gen_txt.name == "seq_gen11_moe_soft.txt"
    assert p.prev_features_csv.name == "features_gen10_moe_soft.csv"
    assert p.prev_labels_csv.name == "labels_gen10_moe_soft.csv"
    assert p.prev_seq_gen_txt.name == "seq_gen10_moe_soft.txt"


def test_skip_data_prep_relaxes_prev_file_check():
    """With skip_data_prep the iteration>0 prev-CSV existence check is bypassed,
    so a seeded run doesn't require iteration_{N-1} data on disk."""
    cfg_seeded = ALConfig(model="hps_urry", iteration=10, front="upper", skip_data_prep=True)
    cfg_seeded.validate()  # must not raise even though prev CSVs don't exist
    cfg_normal = ALConfig(model="hps_urry", iteration=10, front="upper")
    with pytest.raises(FileNotFoundError):
        cfg_normal.validate()


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
