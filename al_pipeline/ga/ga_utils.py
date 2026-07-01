from __future__ import annotations

from typing import Any
from pathlib import Path
import json
import pandas as pd
import numpy as np
import torch
import gpytorch
import random


from al_pipeline.core.config import ALConfig
from al_pipeline.training.ml_models import MultitaskGPRegressionModel, GPRegressionModel 
from al_pipeline.data_prep.data_loading import convert_and_normalize_features
# from al_pipeline.

def load_normalization_stats(file_path: str | Path) -> dict:
    with open(file_path, "r") as f:
        return json.load(f)
    
def seed_everything(*, seed_base: int, iteration: int, seq_id: int, cand_id: int) -> int:
   # one deterministic seed per (iter, seq, cand)
   seed = (seed_base * 1_000_000) + (iteration * 10_000) + (seq_id * 100) + cand_id
   seed = seed % (2**32 - 1)
   random.seed(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   torch.use_deterministic_algorithms(False)
   return seed


atm_types = ['A',
 'C',
 'D',
 'E',
 'F',
 'G',
 'H',
 'I',
 'K',
 'L',
 'M',
 'N',
 'P',
 'Q',
 'R',
 'S',
 'T',
 'V',
 'W',
 'Y']

# convert string to a numpy array
def AA2num(S, atm_types=atm_types):
    # S is a string list
    X = []
    for i in S:
        X.append(atm_types.index(i))
    return np.array(X)

def back_AA(X, atm_types=atm_types):
    X = np.asarray(X) 
    AA_str=[]
    for i in range(X.shape[0]):
        AA_str.append(atm_types[int(X[i])])
    return ''.join(AA_str)


def load_models(cfg: ALConfig, *, temp: bool, device: str | torch.device = "cpu") -> dict[str, Any]:
    """Top-level loader used by GA / augmentation for the global GPR path."""
    t = cfg.train_model_type
    if t == "gpr_multitask":
        return load_gpr_multitask(cfg, temp=temp, device=device)
    if t == "gpr_singletask":
        return load_gpr_singletask(cfg, temp=temp, device=device)
    if t == "moe":
        raise ValueError(
            "load_models is for global GPR only; use load_moe_bundle for "
            "train_model_type='moe' (no model_bundle dict shape — MoEBundle "
            "is passed to make_surrogate via moe_bundle=...)"
        )
    if t == "dnn":
        raise NotImplementedError("DNN loader not implemented yet.")
    raise ValueError(f"Unknown train_model_type={t}")


def load_moe_bundle(cfg: ALConfig, *, temp: bool = False):
    """
    Load the per-iter MoE artifacts written by `train_moe_from_config`.

    Parameters
    ----------
    temp : bool
        False -> base checkpoints from the iter's training run.
        True  -> the augmented checkpoints written by kriging-believer during
                 batch generation (accumulate synthesized children across
                 seq_ids in the current batch).

    Returns
    -------
    MoEBundle
        Validated bundle with the RF + PS + nonPS experts. The bundle's
        metadata is checked against the iter / transform / model_name in cfg
        — stale checkpoints surface as ValueError here rather than as silent
        wrong predictions downstream.
    """
    from al_pipeline.surrogates import MoEBundle
    p = cfg.paths
    return MoEBundle.from_checkpoints(
        str(p.moe_rf_bundle(temp=temp)),
        str(p.moe_ps_chkpt(temp=temp)),
        str(p.moe_nonps_chkpt(temp=temp)),
        str(p.features_csv),
        str(p.labels_csv),
        expected_transform=cfg.transform,
        expected_label_scaler_scope="all",
        expected_model_name=cfg.model,
        expected_iter=cfg.iteration,
    )


def load_gpr_multitask(cfg: ALConfig, *, temp: bool, device: str | torch.device = "cpu") -> dict[str, Any]:
    """
    Loads multitask GPR + likelihood, instantiated with normalized training tensors.
    Returns a dict.
    """
    p = cfg.paths

    X = torch.tensor(pd.read_csv(p.features_norm_csv).values, dtype=torch.float32, device=device)
    y = torch.tensor(pd.read_csv(p.labels_norm_csv).values, dtype=torch.float32, device=device)

    ckpt = torch.load(p.gpr_multitask_chkpt(temp=temp), map_location=device)

    num_tasks = y.shape[1]
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks).to(device)
    model = MultitaskGPRegressionModel(X, y, likelihood, num_tasks=num_tasks).to(device)

    model.load_state_dict(ckpt["model"])
    if "likelihood" in ckpt:
        likelihood.load_state_dict(ckpt["likelihood"])

    model.eval()
    likelihood.eval()

    return {"model": model, "likelihood": likelihood, "X_train": X, "y_train": y}


def load_gpr_singletask(cfg: ALConfig, *, temp: bool, device: str | torch.device = "cpu") -> dict[str, Any]:
    """
    Loads per-objective single-task GPRs.
    """
    p = cfg.paths

    X = torch.tensor(pd.read_csv(p.features_norm_csv).values, dtype=torch.float32, device=device)

    models: dict[str, Any] = {}
    likelihoods: dict[str, Any] = {}
    y_trains: dict[str, Any] = {}

    # TODO: decide whether to use cfg.obj1/cfg.obj2 or a list like cfg.objectives
    for label in (cfg.obj1, cfg.obj2):
       
        y_path = p.labels_csv.with_stem(p.labels_csv.stem + f"_{label}_NORM_{p.tag}")
        y = torch.tensor(pd.read_csv(y_path).values, dtype=torch.float32, device=device).flatten()

        ckpt_path = p.gpr_singletask_chkpt([label], temp=temp)[0]
        ckpt = torch.load(ckpt_path, map_location=device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model = GPRegressionModel(X, y, likelihood).to(device)

        model.load_state_dict(ckpt["model"])
        if "likelihood" in ckpt:
            likelihood.load_state_dict(ckpt["likelihood"])

        model.eval()
        likelihood.eval()

        models[label] = model
        likelihoods[label] = likelihood
        y_trains[label] = y

    return {"models": models, "likelihoods": likelihoods, "X_train": X, "y_train": y_trains}


def load_front(cfg: ALConfig, seq_id: int, log=None):
    """
    Loads the current Pareto front: sequences, raw features, and labels.

    Returns
    -------
    pareto_front : np.ndarray
        Shape (N, 2). Labels in normalized objective space (same z-space the
        EHVI computation operates in).
    pareto_feats_raw_df : pd.DataFrame
        Shape (N, 29). RAW parent features — the surrogate normalizes them
        internally. Lets the epsilon-shift route through the surrogate ABC so
        global GPR and MoE share one code path.
    parent_seqs : list[str]
        The actual amino-acid sequences for each Pareto point.
    """
    p = cfg.paths

    seq_path = p.parent_seqs_temp_txt if seq_id > 1 else p.parent_seqs_txt

    labels_df = pd.read_csv(p.parent_labels_norm_csv)
    feats_raw_df = pd.read_csv(p.parent_features_csv)

    pareto_front = labels_df[[cfg.obj1, cfg.obj2]].to_numpy()

    with open(seq_path, "r") as f:
        parent_seqs = [ln.strip() for ln in f if ln.strip()]

    if len(parent_seqs) != pareto_front.shape[0]:
        raise ValueError("Parents mismatch: sequences and labels have different lengths.")
    if len(feats_raw_df) != pareto_front.shape[0]:
        raise ValueError("Parents mismatch: feats and labels have different lengths.")

    return pareto_front, feats_raw_df, parent_seqs

def alpha(sequences: np.ndarray, propseqs: np.ndarray, seq_id: int) -> np.ndarray:
    """
    Calculate the similarity penalty for a list of sequence features based on the dot product of the sequences
    and previously proposed sequences. (see https://www.science.org/doi/10.1126/sciadv.adj2448)

    Parameters:
    -----------
    sequences: ndarray 
        Sequence features to calculate similarity penalty for
    propseqs: ndarray 
        Features of previously proposed sequence features
    seq_id: int
        ID of the current sequence being generated through the 96 GA runs

    Returns: 
    -----------
    numpy array of similarity penalty values for each sequence
    """
    if seq_id == 1:
        return np.ones(sequences.shape[0])
    else:
        xidotxk = np.matmul(sequences, propseqs.T) # matrix with entries x_i dot x_k
        magnitude_seq = np.sqrt(np.sum(sequences**2, axis=1)) # magnitude of candidates
        magnitude_propseq = np.sqrt(np.sum(propseqs**2, axis=1)) # magnitude of previously proposed

        full_matrix = 0.5 * (1.0 - xidotxk / np.outer(magnitude_seq, magnitude_propseq))
        
        # Replace any zero entries with a small positive value to avoid division by zero
        full_matrix = np.where(full_matrix == 0, 1e-6, full_matrix) ** (-1)

        return (seq_id - 1)/np.sum(full_matrix, axis=1) # similarity penalty per candidate,
        

# Used when we apply the cosine similarity penalty in GA
def load_previous_children_as_feats(
    cfg,
    seq_id: int,
    featurizer,
    normalization_stats: dict,
) -> np.ndarray:
    """
    Returns normalized feature matrix of previous selected children (shape [seq_id-1, nfeat])
    or empty array if seq_id == 1.
    """
    if seq_id <= 1:
        return np.empty((0, 0), dtype=np.float32)

    p = cfg.paths
    prev = []
    for i in range(1, seq_id):
        child_file = p.ga_children_dir / f"seq_child_{i}.txt"
        with open(child_file, "r") as f:
            prev.append(f.readline().strip())

    raw = [np.asarray(featurizer.featurize(s), dtype=np.float64) for s in prev]

    X = np.vstack(raw)
    Xn = convert_and_normalize_features(X, train=False, stats=normalization_stats)  # shape [K,29]
    return np.asarray(Xn, dtype=np.float32)

def make_epsilon_shifted_front(
    cfg,
    pareto_front: np.ndarray,           # shape [N,2] in normalized objective space
    pareto_feats_raw_df: "pd.DataFrame",  # shape [N,29] RAW features
    surrogate,
) -> tuple[np.ndarray, tuple[float, float] | None]:
    """
    Returns (pareto_input, eps_tuple_or_None). pareto_input is what we feed
    into front augmentation.

    Routes through the `Surrogate` ABC so global GPR and MoE share one
    implementation: each surrogate normalizes raw features its own way and
    returns marginal stds via `predict_pool(...).stds`. The shift direction
    flips with `cfg.front` (upper / lower) to push the front "outward" in the
    less explored direction.
    """
    if cfg.ehvi_variant != "epsilon":
        return pareto_front.copy(), None

    epsilon_scale = cfg.epsilon_scale
    pool = surrogate.predict_pool(pareto_feats_raw_df)
    std = pool.stds  # (N, 2) in normalized objective space

    sigma_bar = np.mean(std, axis=0)  # (2,)
    sign = cfg.epsilon_scale if cfg.front == "upper" else -1 * cfg.epsilon_scale
    eps = sign * sigma_bar * epsilon_scale  # (2,)

    pareto_input = pareto_front.copy()
    pareto_input[:, 0] += eps[0]
    pareto_input[:, 1] += eps[1]
    return pareto_input, (float(eps[0]), float(eps[1]))


def select_best_sequence(cfg: ALConfig, seq_id, log=None) -> None:
    """
    Selecting the best sequence from a set of parallel, independent genetic algorithm runs.
    """
    cand_dir = cfg.paths.ga_candidates_dir
    out_path = cfg.paths.ga_children_dir / f"seq_child_{seq_id}.txt"


    best_fitness = float('inf')  # Assuming we minimize the fitness
    best_sequence: str | None = None
    n_seen = 0
    n_skipped = 0

    for path in cand_dir.glob("*.txt"):
        n_seen += 1
        try:
            lines = path.read_text().splitlines()
            if len(lines) < 2:
                raise ValueError("expected at least 2 lines (sequence, fitness)")
            seq = lines[0].strip()
            fit = float(lines[1].strip())
            if fit < best_fitness:
                best_fitness = fit
                best_sequence = seq
        except Exception as e:
            n_skipped += 1
            if log:
                log.warning(f"Skipping malformed candidate file {path.name}: {e}")

    if best_sequence is None:
        msg = f"No valid candidate files found in {cand_dir} (seen={n_seen}, skipped={n_skipped})."
        if log:
            log.error(msg)
        raise RuntimeError(msg)

    with out_path.open("w") as f:
        f.write(best_sequence + "\n")

    if log:
        log.info(f"Sequence {seq_id} saved to {out_path}")
        log.info(f"Best fitness (min): {best_fitness} from {n_seen} files (skipped {n_skipped}).")

    csv_path = cfg.paths.ga_children_dir / f'ehvi_values_{cfg.paths.tag}.csv'
    
    new_row = pd.DataFrame({'Seq_ID': [seq_id], 'Best_Sequence': [best_sequence], 'EHVI': [-1*best_fitness]})

    if seq_id == 1:
        # created csv file for the first sequence
        new_row.to_csv(csv_path, index=False)
    else:
        if csv_path.exists():
            existing_df = pd.read_csv(csv_path)
            updated_df = pd.concat([existing_df, new_row], ignore_index=True)
            updated_df.to_csv(csv_path, index=False)
        else:
            raise RuntimeError(f"Error: {csv_path} does not exist. Creating a new file.")
    

