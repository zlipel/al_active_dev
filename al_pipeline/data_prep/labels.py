from __future__ import annotations

import pandas as pd

from al_pipeline.core.config import ALConfig

def generate_labels(cfg: ALConfig, log=None) -> None:

    p           = cfg.paths
    model_name  = cfg.model.lower()
    iteration   = cfg.iteration

    eos_df  = pd.read_csv(str(p.eos_csv))
    diff_df = pd.read_csv(str(p.diff_csv))

    # extract the relevant columns from eos_df and diff_df
    # from eos_df, get columns 'density', 'density_std', 'exp_density', 'exp_density_std'
    # NOTE: changed to cfg.obj1 and cfg.obj2 for generality
    # TODO: change to general list of primary and auxiliary objectives
    eos_df_subset = eos_df[[cfg.aux1_obj1, f"{cfg.aux1_obj1}_std", cfg.obj1, f"{cfg.obj1}_std"]]
   
    # from diff_df, get columns 'diff', 'diff_std', 
    diff_df_subset = diff_df[[cfg.obj2, f"{cfg.obj2}_std"]]

    required_eos = [cfg.aux1_obj1, f"{cfg.aux1_obj1}_std", cfg.obj1, f"{cfg.obj1}_std"]
    required_diff = [cfg.obj2, f"{cfg.obj2}_std"]
    missing = [c for c in required_eos if c not in eos_df.columns] + [c for c in required_diff if c not in diff_df.columns]
    if missing:
        raise KeyError(f"Missing columns in label CSVs: {missing}")

    # merge the two dataframes on the index, keeping columns from both
    merged_df = pd.concat([eos_df_subset, diff_df_subset], axis=1)

    # add column named generation that is the iteration number (first column)
    merged_df.insert(0, 'generation', iteration)

    # If the iteration is 0, thee become the next labels csv. Otherwise, concatenate with previous labels.
    if iteration == 0:
        merged_df.to_csv(str(p.labels_csv), index=False)
        if log:
            log.info(f"Generated labels for iteration {iteration} and saved to {p.labels_csv}")
    else:
        try:
            prev_labels = pd.read_csv(str(p.prev_labels_csv))
            df_combined = pd.concat([prev_labels, merged_df], ignore_index=True)
            df_combined.to_csv(str(p.labels_csv), index=False)
            if log:
                log.info(f"Generated labels for iteration {iteration}, combined with previous labels, and saved to {p.labels_csv}")
        except Exception as e:
            if log:
                log.exception(f"Couldn't generate labels for iteration {iteration}: {e}")
            else:
                print(f"Couldn't generate labels for iteration {iteration}: {e}")


