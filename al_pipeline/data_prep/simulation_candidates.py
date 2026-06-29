# al_pipeline/data_prep/simulation_candidates.py
from __future__ import annotations

from pathlib import Path
import shutil
from typing import Optional

from al_pipeline.core.config import ALConfig

def generate_simulation_candidates(cfg: ALConfig, log=None) -> Path:
    """
    Collects seq_child_*.txt from the current iteration's GA children folder and
    writes next iteration's simulation_candidates file.

    Parameters:
    ----------
    cfg : ALConfig
      Configuration object.
    log : Optional
      Logger for logging messages.

    Returns:
    -------
    out_file: Path
      Path to the written candidates file (in next_iter_scratch_sim_dir).
    """
    p = cfg.paths

    children_dir = p.ga_children_dir
    if not children_dir.exists():
        raise FileNotFoundError(f"Missing children dir: {children_dir}")

    seqs: list[str] = []
    for seq_id in range(1, cfg.ngen + 1):
        child_file = children_dir / f"seq_child_{seq_id}.txt"
        if not child_file.exists():
            raise FileNotFoundError(f"Missing child file: {child_file}")
        lines = child_file.read_text().strip().splitlines()
        seq = lines[0].strip() if lines else ""
        if not seq:
            raise ValueError(f"Empty sequence in: {child_file}")
        seqs.append(seq)

    out_file = p.next_iter_candidates_file
    out_file.parent.mkdir(parents=True, exist_ok=True)  # usually already exists, harmless

    with out_file.open("w") as f:
        for s in seqs:
            f.write(s + "\n")

    # mirror into eos/diff directories for next iteration
    shutil.copy(out_file, p.next_iter_diff_dir / out_file.name)
    shutil.copy(out_file, p.next_iter_eos_dir / out_file.name)

    if log:
        log.info(f"Wrote simulation candidates: {out_file} ({len(seqs)} sequences)")

    return out_file
