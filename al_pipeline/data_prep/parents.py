from __future__ import annotations

import pandas as pd

from al_pipeline.core.config import ALConfig


def dominates(sol1, sol2, kind = ['max', 'max']):
    """
    Check if sol1 dominates sol2 for maximizing D and minimizing B2.
    """
    obj1 = kind[0]
    obj2 = kind[1]

    if obj1 == 'min':
        if obj2 == 'max':
            return (sol1[0] <= sol2[0] and sol1[1] >= sol2[1]) and (sol1[0] < sol2[0] or sol1[1] > sol2[1])
        elif obj2 == 'min':
            return (sol1[0] <= sol2[0] and sol1[1] <= sol2[1]) and (sol1[0] < sol2[0] or sol1[1] < sol2[1])
    elif obj1 == 'max':
        if obj2 == 'min':
            return (sol1[0] >= sol2[0] and sol1[1] <= sol2[1]) and (sol1[0] > sol2[0] or sol1[1] < sol2[1])
        elif obj2 == 'max':
            return (sol1[0] >= sol2[0] and sol1[1] >= sol2[1]) and (sol1[0] > sol2[0] or sol1[1] > sol2[1])
    else:   
        raise ValueError("Invalid objective types.")

def find_pareto_front(labels, kind = ['max', 'min'], objectives = ['diff', 'exp_density']):
    """
    Identify the non-dominated set based on D (diffusivity) and B2.
    Returns the non-dominated set data and the original indices from the `labels` DataFrame.
    """
    # Copy relevant columns and add original indices
    data = labels[objectives].copy()
    data['original_index'] = labels.index  # Store original indices
    
    # Drop rows where 'diff' is NaN
    data = data.dropna(subset=objectives).reset_index(drop=True)

    # Extract the objectives as a list of solutions for the Pareto front search
    solutions = data[objectives].values.tolist()  #  Convert to list of lists

    non_dominated_set = []
    non_dominated_indices = []

    for i in range(len(solutions)):
        is_dominated = False
        for j in range(len(solutions)):
            if i != j and dominates(solutions[j], solutions[i], kind=kind):  # Check if solution j dominates solution i
                is_dominated = True
                break
        
        if not is_dominated:
            non_dominated_set.append(solutions[i])
            non_dominated_indices.append(data['original_index'].iloc[i])  # Get original index
 
    # Create a DataFrame for the non-dominated set
    non_dominated_df = pd.DataFrame(non_dominated_set, columns=objectives)
    


    return non_dominated_df.reset_index(drop=True), non_dominated_indices


def get_parents(cfg: ALConfig, log=None, stage: str = "base") -> None:
    """Generate normalized parent features, labels, and sequences based on the Pareto front."""
    p = cfg.paths
    obj1, obj2 = cfg.obj1, cfg.obj2
    tag = p.tag

    # Always use normalized generation data for GA + augmentation workflows
    feats_path  = p.features_norm_csv

    if cfg.train_model_type == "gpr_multitask":
        labels_path = p.labels_norm_csv
    elif cfg.train_model_type == "gpr_singletask":
        labels_path = []
        for obj in (obj1, obj2):
            lbl_path = p.labels_csv.with_stem(p.labels_csv.stem + f"_{obj}_NORM_{tag}")
            if not lbl_path.exists():
                raise FileNotFoundError(f"Expected label file for single-task GPR not found: {lbl_path}")
            labels_path.append(lbl_path)
    else:
        raise ValueError(f"Unknown model_train_type: {cfg.model_train_type} not implemented yet.")

    # Select sequences + outputs by stage
    if stage == "base":
        seq_file           = p.seq_gen_txt
        pareto_feats_path  = p.parent_features_norm_csv
        pareto_labels_path = p.parent_labels_norm_csv
        pareto_seq_path    = p.parent_seqs_txt
    elif stage == "temp":
        seq_file           = p.seq_gen_temp_txt
        pareto_feats_path  = p.parent_features_norm_csv
        pareto_labels_path = p.parent_labels_norm_csv
        pareto_seq_path    = p.parent_seqs_temp_txt
    else:
        raise ValueError("stage must be 'base' or 'temp'")

    if cfg.front == "upper":
        kind = ["max", "max"]
    elif cfg.front == "lower":
        kind = ["min", "min"]
    else:
        raise ValueError("Invalid front type. Use 'upper' or 'lower'.")

    objectives = [obj1, obj2]

    features = pd.read_csv(str(feats_path))

    if cfg.train_model_type == "gpr_multitask":
        labels   = pd.read_csv(str(labels_path))
    elif cfg.train_model_type == "gpr_singletask":
        label_dfs = []
        for lbl_path in labels_path:
            label_dfs.append(pd.read_csv(lbl_path))
        labels = pd.concat(label_dfs, axis=1)
        labels.columns = [obj1, obj2]

    _, indices = find_pareto_front(labels, kind=kind, objectives=objectives)

    with open(str(seq_file), "r") as f:
        all_sequences = [line.strip() for line in f]

    if indices and max(indices) >= len(all_sequences):
        raise IndexError("Pareto indices exceed sequence file length; labels/features no longer align with sequences.")

    sequences = [all_sequences[i] for i in indices]

    pareto_features = features.iloc[indices].reset_index(drop=True)
    pareto_labels   = labels.iloc[indices].reset_index(drop=True)

    pareto_features.to_csv(pareto_feats_path, index=False)
    pareto_labels.to_csv(pareto_labels_path, index=False)

    with open(pareto_seq_path, "w") as f:
        for seq in sequences:
            f.write(seq + "\n")

    msg = f"Saved parents: feats={pareto_feats_path}, labels={pareto_labels_path}, seqs={pareto_seq_path}"
    
    if log:
        log.info(msg) 
    else:
        print(msg, flush=True)
