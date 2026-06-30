# al_pipeline/ga/run_ga.py
from __future__ import annotations

from pathlib import Path
from typing import Literal, Callable, Optional

import numpy as np
from pygmo import hypervolume

import al_pipeline.featurization.sequence_featurizer as sf

from al_pipeline.core.config import ALConfig
import al_pipeline.ga.ga_utils as ga_utils
import al_pipeline.acquisition.ehvi as ehvi
from al_pipeline.data_prep.data_loading import convert_and_normalize_features
from al_pipeline.ga.geneticalgorithm_m2 import geneticalgorithm_batch as GA
from al_pipeline.surrogates import Surrogate, make_surrogate



def save_cand_sequence(sequence, fitness, output_folder, cand_id):
    """
    Save the candidate sequence as a text file with the specified candidate ID.
    
    Parameters:
    -----------
        sequence: List[np.ndarray]
            The generated candidate sequence.
        fitness: List[float]
            The fitness value of the candidate sequence.
        output_folder: Path
            Folder where the sequence should be saved.
        cand_id: int
            Unique ID for each candidate (used in the filename).
    """
    seq_file = output_folder / f"seq_cand_{cand_id}.txt"

    # Save sequence to the file
    with seq_file.open("w") as f:
        f.write(ga_utils.back_AA(sequence) + '\n')
        f.write(str(fitness) + '\n')


def _init_pop(parent_seqs: list[str]) -> list[list[int]]:
    return [list(ga_utils.AA2num(s)) for s in parent_seqs]


# analytic ehvi
def _fitness_batch_analytic(
    sequences: list[list[int]],
    *,
    cfg: ALConfig,
    seq_id: int,
    featurizer,
    normalization_stats: dict,
    augmented_front: np.ndarray,  # [N,2] in normalized space (after epsilon shift + front augmentation)
    propseq_arr: np.ndarray,      # [K,nfeat] normalized previous selected children (or empty if not similarity penalty)
    surrogate: Surrogate,
) -> list[float]:
    """
    GA minimizes. We return negative EHVI since the genetic algorithm minimizes but acquisition is maximized (+ optional similarity penalty).
    """
    # ints -> strings -> raw featurized DataFrame
    seq_strs = [ga_utils.back_AA(np.asarray(s, dtype=int)) for s in sequences]
    X_raw_df = featurizer.featurize_many(seq_strs).astype(np.float32)

    # Surrogate consumes RAW features; it normalizes internally with whatever
    # stats it needs (global stats for GlobalGPRSurrogate, per-expert for MoE).
    pool = surrogate.predict_pool(X_raw_df)
    pred1, pred2 = pool.means[:, 0], pool.means[:, 1]
    std1, std2 = pool.stds[:, 0], pool.stds[:, 1]

    # invert sign for upper to change to minimization space
    if cfg.front == "upper":
        vals = ehvi.ehvi_analytic(-pred1, std1, -pred2, std2, augmented_front)
    elif cfg.front == "lower":
        vals = ehvi.ehvi_analytic(pred1, std1, pred2, std2, augmented_front)
    else:
        raise ValueError("cfg.front must be 'upper' or 'lower'")

    vals = np.asarray(vals, dtype=np.float32).reshape(-1)

    # similarity penalty if enabled. The penalty compares candidate features
    # against previously selected children — both in *globally normalized*
    # space, since `propseq_arr` is stored that way. Done lazily so callers
    # that don't use similarity never pay the normalize cost.
    if cfg.exploration_strategy == "similarity_penalty":
        if propseq_arr.size == 0:
            alpha_pen = 1.0
            fit = -vals * alpha_pen
        else:
            Xn = convert_and_normalize_features(
                X_raw_df.to_numpy(dtype=np.float32), train=False, stats=normalization_stats,
            )
            alpha_pen = ga_utils.alpha(np.asarray(Xn, dtype=np.float32), propseq_arr, seq_id)
            fit = -vals * np.asarray(alpha_pen, dtype=np.float32)
    else:
        fit = -vals

    fit = np.where(np.isfinite(fit), fit, 1e9)
    return fit.tolist()


def _fitness_batch_mc(
    sequences: list[list[int]],
    *,
    cfg: ALConfig,
    seq_id: int,
    featurizer,
    normalization_stats: dict,
    pareto_front: np.ndarray,  # [N,2] in normalized objective space + minimization
    ref_point: np.ndarray,  # [2] in normalized objective space + minimization
    base_hv: float,
    propseq_arr: np.ndarray,
    surrogate: Surrogate,
) -> list[float]:
    """
    Does same stuff as analytic but uses Monte Carlo sampling to esitmate the hypervolume.
     
    Parameters:
    -----------
    sequences : list[list[int]]
        List of candidate sequences represented as lists of integers.
    cfg : ALConfig
        Configuration object containing settings for the active learning process.
    seq_id : int
        The current sequence ID being processed.
    featurizer : object
        An object that provides a method to featurize sequences.
    normalization_stats : dict
        Statistics used for normalizing features.
    augmented_front : np.ndarray
        The augmented Pareto front in normalized objective space (shape [N, 2]).
    propseq_arr : np.ndarray
        Array of previously proposed sequences in normalized feature space.
    model_bundle : dict
        A dictionary containing the trained model(s) for prediction.    
    
    Returns:
    --------
    list[float]
        A list of fitness values for each candidate sequence, where lower values indicate better fitness (minimization).
    """
    if not surrogate.supports_joint_sampling:
        raise ValueError(
            "MC-EHVI requires a surrogate that supports joint sampling across "
            "objectives. The current surrogate does not — use gpr_multitask "
            "(or MoE) for MC mode."
        )

    seq_strs = [ga_utils.back_AA(np.asarray(s, dtype=int)) for s in sequences]
    X_raw_df = featurizer.featurize_many(seq_strs).astype(np.float32)

    # Build the joint posterior ONCE for this candidate batch; the MC inner
    # loop samples from it in chunks until the std-error tolerance is hit.
    pool = surrogate.predict_pool(X_raw_df)

    ehvi_vals = ehvi.monte_carlo_ehvi_batch(
        pool,
        pareto_front.copy(),
        ref_point,
        base_hv,
        min_samples=cfg.mc_min_samples,
        max_samples=cfg.mc_max_samples,
        chunk_size=cfg.mc_chunk_size,
        stderr_tol=cfg.mc_stderr_tol,
        front=cfg.front,
    )
    ehvi_vals = np.asarray(ehvi_vals, dtype=np.float32).reshape(-1)

    if cfg.exploration_strategy == "similarity_penalty":
        if propseq_arr.size == 0:
            fit = -ehvi_vals
        else:
            Xn = convert_and_normalize_features(
                X_raw_df.to_numpy(dtype=np.float32), train=False, stats=normalization_stats,
            )
            alpha_pen = ga_utils.alpha(np.asarray(Xn, dtype=np.float32), propseq_arr, seq_id)
            fit = -ehvi_vals * np.asarray(alpha_pen, dtype=np.float32)
    else:
        fit = -ehvi_vals

    fit = np.where(np.isfinite(fit), fit, 1e9)
    return fit.tolist()



def run_one_candidate(
    cfg: ALConfig,
    cand_id: int,
    seq_id: int,
) -> None:
    """
    Runs one GA instance and saves the best candidate for this cand_id.
    No augmentation, no selection of global best.
    """

    temp = seq_id > 1

    # seeding (per-cand reproducibility)
    ga_utils.seed_everything(seed_base=cfg.seed_base, iteration=cfg.iteration, seq_id=seq_id, cand_id=cand_id)

    featurizer = sf.SequenceFeaturizer(model_name=cfg.model.lower(), db_path=cfg.db_path)
    model_bundle = ga_utils.load_models(cfg=cfg, temp = temp, device='cpu')
    normalization_stats = ga_utils.load_normalization_stats(cfg.paths.norm_stats)
    # GA pipeline only wires the global-GPR path here; MoE goes through the
    # CLI in feat/moe-cli-al, which constructs its own MoEBundle.
    surrogate = make_surrogate(
        cfg,
        model_bundle=model_bundle,
        normalization_stats=normalization_stats,
    )

    pareto_front, pareto_feats, parent_seqs = ga_utils.load_front(cfg=cfg, seq_id=seq_id)

    # init population from parent sequences
    init_pop = _init_pop(parent_seqs)

    # previous selected children as normalized feats (only needed if similarity penalty)
    if cfg.exploration_strategy == "similarity_penalty":
        propseq_arr = ga_utils.load_previous_children_as_feats(
            cfg=cfg, seq_id=seq_id, featurizer=featurizer, normalization_stats=normalization_stats
        )
        if propseq_arr.size == 0:
            propseq_arr = np.empty((0, 0), dtype=np.float32)
    else:
        propseq_arr = np.empty((0, 0), dtype=np.float32)

    # epsilon-shift (if enabled) and front augmentation
    pareto_input, eps = ga_utils.make_epsilon_shifted_front(
        cfg=cfg,
        pareto_front=pareto_front,
        pareto_feats=pareto_feats,
        model_bundle=model_bundle,
    )
    

    # choose fitness function variant
    if cfg.mc_ehvi:
        augmented_front, ref_point = ehvi.front_augmentation(
            pareto_input,
            cfg.front,
            ref_mode = cfg.ref_point_mode,
            frac = cfg.ref_point_frac,
            tau = cfg.ref_point_tau,
            cap_frac = cfg.ref_point_cap,
            big = 1e6,
            return_ref = cfg.mc_ehvi,
            mc_mode = cfg.mc_ehvi
        )
        base_hv = hypervolume(augmented_front).compute(ref_point)
        # We use the base_hv for the monte carlo ehvi
        fitness_fn: Callable[[list[list[int]]], list[float]] = lambda seqs: _fitness_batch_mc(
            seqs,
            cfg=cfg,
            seq_id=seq_id,
            featurizer=featurizer,
            normalization_stats=normalization_stats,
            pareto_front=augmented_front,
            ref_point=ref_point,
            base_hv=base_hv,
            propseq_arr=propseq_arr,
            surrogate=surrogate,
        )
    else:
        augmented_front = ehvi.front_augmentation(
            pareto_input,
            cfg.front,
            ref_mode = cfg.ref_point_mode,
            frac = cfg.ref_point_frac,
            tau = cfg.ref_point_tau,
            cap_frac = cfg.ref_point_cap,
            big = 1e6,
            return_ref = cfg.mc_ehvi,
            mc_mode = cfg.mc_ehvi
        )
        fitness_fn = lambda seqs: _fitness_batch_analytic(
            seqs,
            cfg=cfg,
            seq_id=seq_id,
            featurizer=featurizer,
            normalization_stats=normalization_stats,
            augmented_front=augmented_front,
            propseq_arr=propseq_arr,
            surrogate=surrogate,
        )

    # GA parameters (keep in cfg as scalars; no dict in cfg)
    ga_params = {
       'max_num_iteration': cfg.ga_max_iter,
                   'population_size': len(init_pop),
                   'mutation_probability': cfg.ga_mutation_prob,
                   'elit_ratio': cfg.ga_elit_ratio,
                   'crossover_probability': cfg.ga_crossover_prob,
                   'deletion_probability': cfg.ga_deletion_prob,
                   'growth_probability': cfg.ga_growth_prob,
                   'parents_portion': cfg.ga_parents_portion,
                   'crossover_type': 'uniform',
                   'max_iteration_without_improv': cfg.ga_max_no_improv,
                   'maxLen': cfg.ga_Lmax,
                     'minLen': cfg.ga_Lmin,
    }

    ga = GA(
        function=fitness_fn,
        algorithm_parameters=ga_params,
        convergence_curve=False,
        progress_bar=False,
    )
    
    # run the genetic algorithm
    ga.run(init_pop=init_pop)

    best_sequence, best_fit = ga.best_variable, ga.best_function

    save_cand_sequence(
        sequence=best_sequence,
        fitness=best_fit,
        output_folder=cfg.paths.ga_candidates_dir,
        cand_id=cand_id,
    )
