# src/al_pipeline/cli/master.py
from __future__ import annotations

import time
import os

from .child import run_child

from al_pipeline.core.config  import ALConfig
from al_pipeline.core.logging import get_master_logger, get_child_logger, get_log_paths

from al_pipeline.data_prep.features              import generate_features
from al_pipeline.data_prep.labels                import generate_labels
from al_pipeline.data_prep.parents               import get_parents
from al_pipeline.data_prep.simulation_candidates import generate_simulation_candidates  

from al_pipeline.training.kfold_training import train_from_config


def main() -> None:
    cfg = ALConfig.from_cli()

    lp = get_log_paths(cfg)
    lp.master_log.unlink(missing_ok=True)
    for f in lp.child_log_dir.glob("child_*.log"):
        f.unlink(missing_ok=True)
    log = get_master_logger(cfg, also_stdout=True)

    p   = cfg.ensure()

    log.info("MASTER start")

    #### Generate features and labels from simulations ####
    try:
        log.info("Phase : Data preparation .... initiation")
        generate_features(cfg, log=log)
        generate_labels(cfg, log=log)  
        log.info("Phase : Data preparation .... completed")
    except Exception as e:
        log.exception(f"Data preparation failed: {e}")
        raise 


    #### Train the ML Surrogate models ####

    try:
        log.info("Phase : Model training .... initiation")
        train_from_config(cfg, log=log)
        log.info("Phase : Model training .... completed")
    except Exception as e:
        log.exception(f"Model training failed: {e}")
        raise 
    
    # ensure base normalized data exists after training
    assert p.features_norm_csv.exists(), f"Missing {p.features_norm_csv}"

    if cfg.train_model_type == "gpr_multitask":
        assert p.labels_norm_csv.exists(), f"Missing {p.labels_norm_csv}"
    elif cfg.train_model_type == "gpr_singletask":
        for obj in (cfg.obj1, cfg.obj2):
            lbl_path = p.labels_csv.with_stem(p.labels_csv.stem + f"_{obj}_NORM_{p.tag}")
            assert lbl_path.exists(), f"Missing {lbl_path}"
    else:
        raise ValueError(f"Unknown train_model_type: {cfg.train_model_type} not implemented yet.")
    
    #### Get parents for next generation ####
    try:
        log.info("Phase : Parent selection .... initiation")
        get_parents(cfg, log=log, stage="base")
        log.info("Phase : Parent selection .... completed")
    except Exception as e:
        log.exception(f"Parent selection failed: {e}")
        raise 
    
    # ensure parents exist after selection
    assert p.parent_seqs_txt.exists(), f"Missing {p.parent_seqs_txt}"
    assert p.parent_features_norm_csv.exists(), f"Missing {p.parent_features_norm_csv}"
    assert p.parent_labels_norm_csv.exists(), f"Missing {p.parent_labels_norm_csv}"

    #### Launch child processes for GA candidate evaluations ####
    log.info("Launching child processes for GA candidates")

    t_init = time.time()
    slurm_cpus_per_task = os.getenv("SLURM_CPUS_PER_TASK")

    num_workers = min(cfg.ncands, int(slurm_cpus_per_task))

    for seq_id in range(1, cfg.ngen + 1):

        #### Get child logger ####
        child_log = get_child_logger(cfg, seq_id, also_stdout=False)

        child_log.info(f"Starting children for seq_id {seq_id}")
        try:
            log.info(f"Launching child processes for seq_id {seq_id}...")
            run_child(cfg, num_workers=num_workers, seq_id=seq_id, log=child_log)
            log.info(f"Child processes for seq_id {seq_id} completed.")
        except Exception as e:
            log.exception(f"Child processes for seq_id {seq_id} failed: {e}")
            raise 
    t_end = time.time()

    elapsed = t_end - t_init

    hrs = int(elapsed // 3600)
    mins = int((elapsed % 3600) // 60)
    secs = int(elapsed % 60)

    log.info(f"All children generated in {hrs}h {mins}m {secs}s.")

    if cfg.acq_test:

        from al_pipeline.data_prep.features import generate_child_features
        log.info("Generating child features csv for acquisition function evaluation...")
        try:
            generate_child_features(cfg, log=log)
            log.info("Child features generation completed.")
        except Exception as e:
            log.exception(f"Child features generation failed: {e}")
            raise


    #### Generate next sim files and so on ####

    try:
        log.info("Phase : Generating simulation candidates .... initiation")
        generate_simulation_candidates(cfg, log=log)
        log.info("Phase : Generating simulation candidates .... completed")
    except Exception as e:
        log.exception(f"Generating simulation candidates failed: {e}")
        raise 

    log.info("MASTER done")

if __name__ == "__main__":
    main()