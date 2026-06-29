from __future__ import annotations

import pandas as pd
import json

from al_pipeline.core.config import ALConfig
from al_pipeline.featurization.sequence_featurizer import SequenceFeaturizer
from al_pipeline.data_prep.data_loading import convert_and_normalize_features


def generate_features(cfg: ALConfig, log=None) -> None:

    p = cfg.paths
    model_name = cfg.model.lower()

    db_path = cfg.db_path
    iter_num = cfg.iteration
    featurizer = SequenceFeaturizer(model_name, str(db_path))

    seq_file = p.seq_gen_txt
    new_seqs = p.eos_dir / f"seq_gen{iter_num}.txt"

    with open(new_seqs, 'r') as f:
        sequences = [line.strip() for line in f]

    if iter_num == 0:
        df = featurizer.featurize_many(sequences)
        df.to_csv(p.features_csv, index=False)
        if log:
            log.info(f"Generated features for iteration {iter_num} and saved to {p.features_csv}")
    else:
        try:
            old_seq_file = p.prev_iter_scratch_dir / f"seq_gen{iter_num - 1}.txt"
            with open(old_seq_file, 'r') as f:
                old_sequences = [line.strip() for line in f]
            
            # combine old and new sequences. 
            sequences = old_sequences + sequences

            # write combined sequences to this generation's sequence file 
            with open(seq_file, 'w') as f:
                for seq in sequences:
                    f.write(seq + '\n')
                    
            df = featurizer.featurize_many(sequences)
            prev_feats = pd.read_csv(p.prev_features_csv)
            combined_df = pd.concat([prev_feats, df], ignore_index=True)
            combined_df.to_csv(p.features_csv, index=False)
            if log:
                log.info(f"Generated features for iteration {iter_num}, combined with previous features, and saved to {p.features_csv}")
        except Exception as e:
            if log:
                log.error(f"Couldn't generate features for iteration {iter_num}: {e}")
                raise e
            else:
                print(f"Couldn't generate features for iteration {iter_num}: {e}")


def generate_child_features(cfg: ALConfig, log=None) -> None:
    p = cfg.paths
    model_name = cfg.model.lower()

    db_path = cfg.db_path
    iter_num = cfg.iteration
    featurizer = SequenceFeaturizer(model_name, str(db_path))

    children_folder = p.ga_children_dir
    if not children_folder.exists():
        if log:
            log.warning(f"Children folder {children_folder} does not exist. Skipping child feature generation.")
        else:
            print(f"Children folder {children_folder} does not exist. Skipping child feature generation.")
        return
    
    # loop through all children files and generate features for each candidate, then combine into one parent features csv
    all_sequences = []

    for child_file in children_folder.glob("seq_child_*.txt"):
        with open(child_file, 'r') as f:
            sequences = [line.strip() for line in f]
            all_sequences.extend(sequences)

    # now, featurize all sequences and save to child features csv
    df = featurizer.featurize_many(all_sequences)
    raw_features = df.values

    df_columns = df.columns.tolist()

    # normalize features using stats 

    with open(p.norm_stats, "r") as f:
        norm_stats = json.load(f)
    normed_feats = convert_and_normalize_features(raw_features, train=False, stats=norm_stats)

    df_normed = pd.DataFrame(normed_feats, columns=df_columns)

    child_csv_file     = p.ga_children_dir / f"child_features.csv"
    child_csv_raw_file = p.ga_children_dir / f"child_features_raw.csv"

    df.to_csv(child_csv_raw_file, index=False)
    df_normed.to_csv(child_csv_file, index=False)

    if log:
        log.info(f"Generated child features for iteration {iter_num} and saved to {child_csv_file}")
        
