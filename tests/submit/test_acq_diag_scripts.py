"""Isolated diagnostic launch tests: no scheduler, GA, or simulation is invoked."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


REPO = Path(__file__).resolve().parents[2]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def isolated(tmp_path):
    repo = tmp_path / "repo"
    (repo / "submit").mkdir(parents=True)
    (repo / "config").mkdir()
    for pattern in ("*acq*.sh", "acq_diag_tools.py"):
        for source in (REPO / "submit").glob(pattern):
            shutil.copy2(source, repo / "submit" / source.name)
    ph, ps, asc = [tmp_path / name for name in ("home_prod", "scratch_prod", "scratch_acq")]
    ah = ph / "ACQ_SWEEP"
    (repo / "config/cluster.env").write_text(
        f'HOME_AL="{ph}"\nSCRATCH_AL="{ps}"\nHOME_AL_ACQ="{ah}"\nSCRATCH_AL_ACQ="{asc}"\n'
        f'CONDA_MODULE=mock\nCONDA_ENV=mock\nCONDA_PREFIX="{tmp_path}/conda"\nDB_PATH="{tmp_path}/db"\n')
    bins = tmp_path / "bin"
    bins.mkdir()
    for name in ("module", "conda"):
        (bins / name).write_text("#!/bin/bash\nexit 0\n")
        (bins / name).chmod(0o755)
    capture = tmp_path / "capture.py"
    capture.write_text("import sys,json,os\nfrom pathlib import Path\n"
                       "Path(os.environ['ACQ_TEST_CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n")
    (bins / "python").write_text(
        '#!/bin/bash\nif [[ "$2" == seeded-master ]]; then\n'
        f'  exec "{sys.executable}" "{capture}" "$@"\nfi\nexec "{sys.executable}" "$@"\n')
    (bins / "python").chmod(0o755)
    (bins / "sbatch").write_text(f'#!/bin/bash\nexec "{sys.executable}" "{capture}" "$@"\n')
    (bins / "sbatch").chmod(0o755)
    env = os.environ.copy()
    env.update(PATH=f"{bins}:{env['PATH']}", SLURM_SUBMIT_DIR=str(repo),
               ACQ_TEST_CAPTURE=str(tmp_path / "captured.json"))
    return dict(repo=repo, ph=ph, ps=ps, ah=ah, asc=asc, env=env, tmp=tmp_path)


def write_source(f, it=1):
    directory = f["ps"] / "MPIPI/GENERATIONS" / f"iteration_{it}"
    directory.mkdir(parents=True)
    n = 120 + 48 * it
    x = pd.DataFrame(np.ones((n, 29)), columns=[f"f{i}" for i in range(28)] + ["length"])
    x["length"] = 20
    x.to_csv(directory / f"features_gen{it}.csv", index=False)
    y = pd.DataFrame({"generation": [0] * 120 + [i for i in range(1, it + 1) for _ in range(48)],
                      "density": np.ones(n), "exp_density": np.ones(n), "diff": np.ones(n)})
    y.to_csv(directory / f"labels_gen{it}.csv", index=False)
    (directory / f"seq_gen{it}.txt").write_text("ACDEFGHIKLMNPQRSTVWY\n" * n)
    return directory


def run(f, script, *args):
    return subprocess.run(["bash", str(f["repo"] / "submit" / script), *args],
                          cwd=f["tmp"], env=f["env"], capture_output=True, text=True)


def stage(f, name="valid", it="1"):
    return run(f, "stage_acq_snapshots.sh", "--model", "MPIPI", "--iters", it, "--exp_name", name)


def test_shell_syntax():
    for name in ("acq_diag_lib.sh", "acq_diag_cell.sh", "stage_acq_snapshots.sh", "run_acq_diag.sh"):
        assert subprocess.run(["bash", "-n", str(REPO / "submit" / name)]).returncode == 0


def test_stage_is_atomic_and_hash_checked(isolated):
    f = isolated
    src = write_source(f)
    original = {p.name: p.read_bytes() for p in src.iterdir()}
    assert stage(f).returncode == 0
    dst = f["asc"] / "valid/_snapshots/MPIPI/iteration_1"
    assert (dst / "snapshot_manifest.json").is_file()
    assert stage(f).returncode == 0
    assert {p.name: p.read_bytes() for p in src.iterdir()} == original
    path = dst / "labels_gen1.csv"
    path.chmod(0o644)
    path.write_text(path.read_text() + "\n")
    assert stage(f).returncode != 0
    r = run(f, "run_acq_diag.sh", "--iters", "1", "--exp_name", "valid", "--dry_run")
    assert r.returncode != 0 and "hash mismatch" in r.stderr


@pytest.mark.parametrize("fault", ["nan", "inf", "missing", "future", "blank"])
def test_invalid_snapshot_never_published(isolated, fault):
    f = isolated
    src = write_source(f)
    path = src / "labels_gen1.csv"
    y = pd.read_csv(path)
    if fault in ("nan", "inf"):
        y.loc[0, "diff"] = np.nan if fault == "nan" else np.inf
    elif fault == "missing":
        y = y.drop(columns="diff")
    elif fault == "future":
        y.loc[0, "generation"] = 2
    else:
        (src / "seq_gen1.txt").write_text((src / "seq_gen1.txt").read_text() + "\n")
    y.to_csv(path, index=False)
    assert stage(f, "invalid").returncode != 0
    assert not (f["asc"] / "invalid/_snapshots/MPIPI/iteration_1").exists()
    assert stage(f, "invalid").returncode != 0


@pytest.mark.parametrize("method,pessimism,strategy", [("kb", False, "kriging_believer"),
                         ("pkb", True, "kriging_believer"), ("independent", False, "standard")])
def test_cell_flags_and_namespaces(isolated, method, pessimism, strategy):
    f = isolated
    write_source(f)
    assert stage(f).returncode == 0
    args = ("--model", "MPIPI", "--iter", "1", "--seed", "12345", "--method", method,
            "--exp_name", "valid", "--pilot")
    result = run(f, "acq_diag_cell.sh", *args)
    assert result.returncode == 0, result.stderr
    argv = json.loads(Path(f["env"]["ACQ_TEST_CAPTURE"]).read_text())
    assert argv[argv.index("--master-seed") + 1] == "271828"
    assert ("--pessimism" in argv) == pessimism
    assert argv[argv.index("--exploration_strategy") + 1] == strategy
    assert "--skip_data_prep" in argv and "--acq_test" in argv and "--holdout_plot" not in argv
    assert "--mc_ehvi" not in argv
    assert argv[argv.index("--ngen") + 1] == "4"
    assert "_upper_pilot" in argv[argv.index("--base_path") + 1]
    assert run(f, "acq_diag_cell.sh", *args).returncode == 2
    assert run(f, "acq_diag_cell.sh", *args, "--front", "lower").returncode == 0


def test_mock_submission_sets_workdir_and_validates_grid(isolated):
    f = isolated
    write_source(f)
    assert stage(f).returncode == 0
    args = ("--iters", "1", "--methods", "kb", "--exp_name", "valid", "--pilot")
    assert run(f, "run_acq_diag.sh", *args).returncode == 0
    argv = json.loads(Path(f["env"]["ACQ_TEST_CAPTURE"]).read_text())
    assert f"--chdir={f['repo']}" in argv
    assert run(f, "run_acq_diag.sh", *args, "--methods", "bad").returncode != 0


def test_symlink_escape_refused(isolated):
    f = isolated
    write_source(f)
    f["asc"].mkdir()
    (f["asc"] / "alias").symlink_to(f["ps"], target_is_directory=True)
    assert stage(f, "alias").returncode != 0


def test_master_seed_reproducible_without_running_training(monkeypatch, tmp_path):
    import random
    import torch
    import al_pipeline.cli.master as master
    tools = load_module(REPO / "submit/acq_diag_tools.py", "acq_tools_test")
    draws = []
    monkeypatch.setattr(sys, "argv", sys.argv[:])

    def stop_before_training():
        draws.append((random.random(), float(np.random.random()), float(torch.rand(()))))
        raise RuntimeError("stop_before_training")

    monkeypatch.setattr(master, "main", stop_before_training)
    args = ["--model", "MPIPI", "--iter", "1", "--front", "upper", "--skip_data_prep",
            "--train_model_type", "gpr_multitask", "--transform", "yeoj", "--ehvi_variant", "epsilon",
            "--exploration_strategy", "kriging_believer", "--obj1", "exp_density", "--obj2", "diff"]
    for ga_seed in (12345, 23456):
        with pytest.raises(RuntimeError, match="stop_before_training"):
            tools.seeded_master(271828, args + ["--seed_base", str(ga_seed)])
    assert draws[0] == draws[1]


def test_report_pairs_and_rejects_unmatched_fits():
    report = load_module(REPO / "utils/acq_diag_report.py", "acq_report_test")
    rows = []
    for method, seed, score in [("kb", 1, 2.), ("pkb", 1, 3.), ("pkb", 2, 100.)]:
        rows.append(dict(model="MPIPI", front="upper", budget="full", iter=1,
                         method=method, seed=seed, vendi=score, base_fit="same", batch_n=24,
                         ncands=96, ga_max_iter=200))
    data = pd.DataFrame(rows)
    with pytest.warns(UserWarning, match="Unpaired"):
        result = report.build_increase(data)
    assert result.iloc[0]["increase_abs"] == 1 and result.iloc[0]["n_pairs"] == 1
    data.loc[data["method"] == "pkb", "base_fit"] = "different"
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="base_fit"):
        report.build_increase(data)


def test_completion_fingerprint_and_report_without_training(monkeypatch, tmp_path):
    import torch
    import al_pipeline.cli.master as master
    from al_pipeline.core.config import ALConfig
    from al_pipeline.diagnostic.batch_diversity import diversity_rows
    tools = load_module(REPO / "submit/acq_diag_tools.py", "acq_tools_completion_test")
    report = load_module(REPO / "utils/acq_diag_report.py", "acq_report_completion_test")
    monkeypatch.setattr(sys, "argv", sys.argv[:])

    def write_fake_completed_outputs():
        cfg = ALConfig.from_cli()
        p = cfg.ensure()
        for name in tools.snapshot_files(cfg.iteration):
            (p.iter_scratch_dir / name).write_text("fixed raw snapshot\n")
        p.norm_stats.write_text('{"fixed": true}')
        torch.save({"model": {"weight": torch.rand(3)}}, p.gpr_multitask_chkpt(temp=False))
        X = np.random.normal(size=(cfg.ngen, 29))
        p.diagnostic_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(X).to_csv(p.diagnostic_dir / f"batch_features_norm_{p.tag}.csv", index=False)
        pd.DataFrame(diversity_rows(X, space="feature")).to_csv(
            p.diagnostic_dir / f"batch_diversity_{p.tag}.csv", index=False)
        p.next_iter_candidates_file.write_text("AAAA\n" * cfg.ngen)
        pd.DataFrame({"Seq_ID": range(1, cfg.ngen + 1), "EHVI": np.ones(cfg.ngen)}).to_csv(
            p.ga_children_dir / f"ehvi_values_{p.tag}.csv", index=False)

    monkeypatch.setattr(master, "main", write_fake_completed_outputs)
    args = ["--model", "MPIPI", "--iter", "1", "--front", "upper", "--skip_data_prep",
            "--train_model_type", "gpr_multitask", "--transform", "yeoj", "--ehvi_variant", "epsilon",
            "--exploration_strategy", "kriging_believer", "--obj1", "exp_density", "--obj2", "diff",
            "--ngen", "4", "--ncands", "8", "--ga_max_iter", "20", "--seed_base", "12345"]
    fingerprints = []
    for method in ("kb", "pkb"):
        cell = f"MPIPI_iter1_seed12345_{method}_upper_pilot"
        home = tmp_path / "home" / "experiment" / cell
        flags = ["--pessimism"] if method == "pkb" else []
        tools.seeded_master(271828, args + ["--base_path", str(home), "--scratch_path", str(tmp_path / cell)] + flags)
        complete = json.loads((home / "completed.json").read_text())
        fingerprints.append(complete["base_fit"])
    assert fingerprints[0] == fingerprints[1]
    cells = report.build_per_cell(tmp_path / "home", "experiment", "MPIPI", budget="pilot")
    result = report.build_increase(cells)
    assert result.iloc[0]["increase_abs"] == 0
