# src/al_pipeline/cli/child.py
from __future__ import annotations

import os
# Hard cap threading so that spawns don't intercommunicate in some way and stay independent 
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


import multiprocessing as mp

from al_pipeline.core.config import ALConfig
from al_pipeline.ga import ga_utils, run_ga
from al_pipeline.ga import augmentation
from al_pipeline.data_prep.parents  import get_parents


def _worker(
    cfg: ALConfig,
    cand_id: int,
    seq_id: int,
) -> None:
    run_ga.run_one_candidate(
            cfg=cfg,
            cand_id=cand_id,
            seq_id=seq_id,
        )

def _init_worker_threads():
    # initialize threading limits in each worker process
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def run_child(cfg, num_workers: int = 96, seq_id: int = 1, log=None) -> None:

    if log:
        log.info(f"Starting child process for seq_id {seq_id} with {num_workers} workers.")

    for f in cfg.paths.ga_candidates_dir.glob("*"):
        if f.is_file():
            f.unlink()
            
    ctx = mp.get_context("spawn")

    # Different candidates for the same seq_id to process in parallel
    tasks = [(cfg, cand_id, seq_id) for cand_id in range(1, cfg.ncands + 1)]

    with ctx.Pool(
        processes=num_workers*2, #TODO: adjust cfg to set oversubscription factor
        initializer=_init_worker_threads,
        maxtasksperchild=1,  # optional: avoids memory growth in long runs
    ) as pool:
        # starmap will raise if any worker errors
        try:
            pool.starmap(_worker, tasks)
        except Exception as e:
            if log:
                log.exception(f"Error in worker processes for seq_id {seq_id}: {e}")
            else:
                print(f"Error in worker processes for seq_id {seq_id}: {e}")
            raise e

    if log:
        log.info(f"All candidates for seq_id {seq_id} have been processed.")

    ga_utils.select_best_sequence(cfg, seq_id, log=log) 

    if cfg.exploration_strategy not in ['standard', 'similarity_penalty']:
        # Do feature/label augmentation for next child
        if log:
            log.info(f"Starting augmentation for seq_id {seq_id}.")
        augmentation.augment(cfg, seq_id=seq_id, pessimism=cfg.pessimism, log=log)
        if log:
            log.info(f"Augmentation for seq_id {seq_id} completed.")

        get_parents(cfg, stage="temp", log=log)

    else:

        if seq_id == 1:
            # copy sequences to temp file for next child since no augmentation
            all_seqs = []
            with open(cfg.paths.parent_seqs_txt, "r") as f:
                for line in f:
                    all_seqs.append(line.strip())
            with open(cfg.paths.parent_seqs_temp_txt, "w") as f:
                for seq in all_seqs:
                    f.write(seq + "\n")
        # Copy parents to temp files for next child 
            # get_parents(cfg, stage="temp", log=log)
            # now copy the gpr files to temp for next child since no augmentation
            
        else:
            # move on, no need to repeat
            pass
    # clean up candidate folder 
    
    cand_dir = cfg.paths.ga_candidates_dir
    for cand_file in cand_dir.glob("*"):
        if cand_file.is_file():
            cand_file.unlink()

    if log:
        log.info(f"Child process for seq_id {seq_id} completed.")