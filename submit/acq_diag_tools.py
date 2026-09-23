"""Validation and deterministic entry point for isolated acquisition diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def snapshot_files(iteration: int) -> list[str]:
    return [f"features_gen{iteration}.csv", f"labels_gen{iteration}.csv",
            f"seq_gen{iteration}.txt"]


def check_paths(args) -> None:
    paths = [Path(p) for p in (args.production_home, args.production_scratch,
                              args.home_root, args.scratch_root)]
    if not all(p.is_absolute() for p in paths):
        raise ValueError("Production and diagnostic roots must be absolute paths")
    ph, ps, ah, asc = [p.resolve() for p in paths]
    if ah == ph or ah.is_relative_to(ps):
        raise ValueError("Diagnostic HOME root aliases a production root")
    if asc == ph or asc.is_relative_to(ps) or asc.is_relative_to(ph):
        raise ValueError("Diagnostic scratch root must be outside production roots")
    if ah == asc or ah.is_relative_to(asc) or asc.is_relative_to(ah):
        raise ValueError("Diagnostic HOME and scratch roots must be distinct")
    if args.target is not None:
        target = Path(args.target)
        if not target.is_absolute() or ".." in target.parts:
            raise ValueError(f"Invalid diagnostic target: {target}")
        target = target.resolve()
        if not (target.is_relative_to(ah) or target.is_relative_to(asc)):
            raise ValueError(f"Target resolves outside diagnostic roots: {target}")
        if target.is_relative_to(ps):
            raise ValueError(f"Target resolves into production scratch: {target}")


def validate_data(directory: Path, iteration: int) -> int:
    import numpy as np
    import pandas as pd

    x = pd.read_csv(directory / f"features_gen{iteration}.csv")
    y = pd.read_csv(directory / f"labels_gen{iteration}.csv")
    seqs = (directory / f"seq_gen{iteration}.txt").read_text().splitlines()
    expected = 120 + 48 * iteration
    if not len(x) == len(y) == len(seqs) == expected:
        raise ValueError(f"Row mismatch: features={len(x)}, labels={len(y)}, "
                         f"sequences={len(seqs)}, expected={expected}")
    if x.shape[1] != 29 or not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("Expected 29 finite numeric raw features")
    required = ["generation", "density", "exp_density", "diff"]
    if missing := sorted(set(required) - set(y.columns)):
        raise ValueError(f"Missing required label columns: {missing}")
    if not np.isfinite(y[required].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite required labels")
    gen = y["generation"].to_numpy(dtype=float)
    if (gen < 0).any() or (gen > iteration).any() or (gen != gen.astype(int)).any():
        raise ValueError("Invalid or future generation labels")
    counts = y.groupby("generation").size().to_dict()
    if counts != {i: 120 if i == 0 else 48 for i in range(iteration + 1)}:
        raise ValueError(f"Unexpected generation counts: {counts}")
    alphabet = set("ACDEFGHIKLMNPQRSTVWY")
    if any(not s or s != s.strip() or not set(s) <= alphabet for s in seqs):
        raise ValueError("Blank, whitespace-containing, or invalid sequence")
    if "length" not in x or not np.array_equal(x["length"].values, [len(s) for s in seqs]):
        raise ValueError("Raw feature lengths do not match sequence rows")
    return expected


def verify_snapshot(directory: Path, iteration: int) -> dict:
    manifest_path = directory / "snapshot_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Incomplete snapshot: missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    expected = snapshot_files(iteration)
    if manifest.get("iteration") != iteration or set(manifest.get("sha256", {})) != set(expected):
        raise ValueError("Snapshot manifest does not match requested iteration/files")
    if manifest.get("n_rows") != 120 + 48 * iteration:
        raise ValueError("Snapshot manifest has an invalid row count")
    for name in expected:
        path = directory / name
        if path.is_symlink() or not path.is_file() or file_hash(path) != manifest["sha256"][name]:
            raise ValueError(f"Snapshot hash mismatch or missing file: {path}")
    return manifest


def stage_snapshot(source: Path, destination: Path, iteration: int, force: bool) -> None:
    if destination.exists():
        if force:
            raise ValueError("Do not replace a frozen experiment snapshot; use a new --exp_name")
        verify_snapshot(destination, iteration)
        print(f"Verified existing frozen snapshot: {destination}")
        return
    for name in snapshot_files(iteration):
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.with_name(destination.name + ".stage-lock")
    lock.mkdir()
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        if destination.exists():
            raise FileExistsError(destination)
        hashes = {}
        for name in snapshot_files(iteration):
            original = source / name
            before = file_hash(original)
            shutil.copyfile(original, temporary / name)
            if file_hash(original) != before or file_hash(temporary / name) != before:
                raise ValueError(f"Source changed during staging: {original}")
            hashes[name] = before
        n = validate_data(temporary, iteration)
        # Detect source edits while a different member of the snapshot was copied.
        if any(file_hash(source / name) != value for name, value in hashes.items()):
            raise ValueError("Source changed during snapshot validation")
        manifest = {"schema": 1, "iteration": iteration, "n_rows": n,
                    "source": str(source.resolve()), "sha256": hashes}
        (temporary / "snapshot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        for path in temporary.iterdir():
            path.chmod(0o444)
        temporary.rename(destination)
        destination.chmod(0o555)
        print(f"Staged and hash-verified {n} rows: {destination}")
    finally:
        if temporary.exists():
            temporary.chmod(0o755)
            shutil.rmtree(temporary)
        lock.rmdir()


def seeded_master(seed: int, argv: list[str]) -> None:
    import random
    import numpy as np
    import pandas as pd
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    from al_pipeline.core.config import ALConfig
    from al_pipeline.cli.master import main

    sys.argv = ["al-master", *(argv[1:] if argv[:1] == ["--"] else argv)]
    cfg = ALConfig.from_cli()
    main()

    # Persist a portable fingerprint of the unchanged base fit, not serialization bytes.
    checkpoint = torch.load(cfg.paths.gpr_multitask_chkpt(temp=False), map_location="cpu", weights_only=True)
    h = hashlib.sha256()
    for name, value in sorted(checkpoint["model"].items()):
        h.update(name.encode())
        h.update(str((value.dtype, tuple(value.shape))).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    raw_hashes = {name: file_hash(cfg.paths.iter_scratch_dir / name)
                  for name in snapshot_files(cfg.iteration)}
    base_fit = {"master_seed": seed, "parameter_sha256": h.hexdigest(),
                "normalization_sha256": file_hash(cfg.paths.norm_stats), "training_sha256": raw_hashes}
    home = cfg.base_path
    (home / "base_fit_fingerprint.json").write_text(json.dumps(base_fit, indent=2) + "\n")

    matrix = cfg.paths.diagnostic_dir / f"batch_features_norm_{cfg.paths.tag}.csv"
    diversity = cfg.paths.diagnostic_dir / f"batch_diversity_{cfg.paths.tag}.csv"
    proposals = cfg.paths.next_iter_candidates_file
    ehvi = cfg.paths.ga_children_dir / f"ehvi_values_{cfg.paths.tag}.csv"
    X = pd.read_csv(matrix).to_numpy(dtype=float)
    scores = pd.read_csv(diversity)
    if X.shape != (cfg.ngen, 29) or not np.isfinite(X).all():
        raise ValueError("Diagnostic batch features incomplete/non-finite")
    if len(proposals.read_text().splitlines()) != cfg.ngen or len(pd.read_csv(ehvi)) != cfg.ngen:
        raise ValueError("Diagnostic proposal/EHVI count does not match requested batch")
    if not np.isfinite(scores[["vendi", "vendi_norm", "mean_distance"]].to_numpy()).all():
        raise ValueError("Non-finite batch diversity score")
    result = {"schema": 1, "model": cfg.model, "iteration": cfg.iteration,
              "front": cfg.front, "ga_seed": cfg.seed_base, "master_seed": seed,
              "ngen": cfg.ngen, "ncands": cfg.ncands, "ga_max_iter": cfg.ga_max_iter,
              "tag": cfg.paths.tag, "base_fit": base_fit}
    (home / "completed.json").write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check-paths")
    for flag in ("production-home", "production-scratch", "home-root", "scratch-root"):
        check.add_argument(f"--{flag}", required=True)
    check.add_argument("--target")
    stage = commands.add_parser("stage")
    stage.add_argument("--source", type=Path, required=True)
    stage.add_argument("--destination", type=Path, required=True)
    stage.add_argument("--iteration", type=int, required=True)
    stage.add_argument("--force", action="store_true")
    verify = commands.add_parser("verify")
    verify.add_argument("--directory", type=Path, required=True)
    verify.add_argument("--iteration", type=int, required=True)
    seeded = commands.add_parser("seeded-master")
    seeded.add_argument("--master-seed", type=int, default=271828)
    seeded.add_argument("pipeline_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == "check-paths":
        check_paths(args)
    elif args.command == "stage":
        stage_snapshot(args.source, args.destination, args.iteration, args.force)
    elif args.command == "verify":
        verify_snapshot(args.directory, args.iteration)
    elif args.command == "seeded-master":
        seeded_master(args.master_seed, args.pipeline_args)


if __name__ == "__main__":
    main()
