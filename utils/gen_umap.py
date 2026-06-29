import json
import numpy as np
import torch
import gpytorch
import pandas as pd
import argparse
from time import time
import os
from TRAINING.gpr_model import GPRegressionModel, MultitaskGPRegressionModel
from geneticalgorithm_m2 import geneticalgorithm_batch as ga
from UTILS.data_preprocessing_gpr import load_dataset
# from utils.ehvi_2d import psi, ehvi_batch
import UTILS.ehvi as ehvi
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import sequence_featurizer as sf
import pickle
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from sklearn.preprocessing import StandardScaler, PowerTransformer

import sys

import utils.plotting as plotting
from utils.pareto_methods import *

def load_gpr_models(model_path, master_path, iteration, model_labels, ehvi_var, explore, seq_id, transform, front):
    """
    Load the GPR models from the specified path.

    Args:
        model_path: Path to the saved GPR models.
        iteration: The iteration number for the active learning process.
        model_labels: List of labels for the GPR models.

    Returns:
        List of GPR models, likelihoods, and scalers (for labels).
    """

    # Check if model file exists

    scalers = []
    labels_total = []


    gpr_file = os.path.join(model_path,f'GPR_iter{iteration}_{ehvi_var}_{explore}_{transform}_{front}.pt')

    # features_file = os.path.join(master_path,f'features_gen{iteration}_TEMP.csv') if seq_id > 1 and explore == 'front_augmentation' \
    #     else os.path.join(master_path,f'features_gen{iteration}.csv')

    features_file = os.path.join(master_path,f'features_gen{iteration}.csv')
    norm_features_file = os.path.join(master_path,f'features_gen{iteration}_NORM_{ehvi_var}_{explore}_{transform}.csv')

    labels_file = os.path.join(master_path,f'labels_gen{iteration}.csv')
    norm_labels_file = os.path.join(master_path,f'labels_gen{iteration}_NORM_{ehvi_var}_{explore}_{transform}.csv')

    # norm_features_file = os.path.join(master_path,f'features_gen{iteration}.csv')
    # norm_labels_file = os.path.join(master_path,f'labels_gen{iteration}.csv')

    # next block is to load the

    for label in model_labels:

        if not os.path.exists(features_file):
            raise FileNotFoundError(f"Features file not found at {features_file}")

        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"Labels file not found at {labels_file}")

        # load the training data
        features, labels = load_dataset(features_file, \
                                                labels_file, \
                                                    label_column=label)
        if label == 'diff' and 'log' in transform:
            # Convert diffusion coefficient to log scale
            labels = np.log(labels + 1e-8)

        if 'yeoj' in transform:
            scaler = PowerTransformer(method='yeo-johnson')
            scaler.fit_transform(labels.reshape(-1,1))
        elif 'log' in transform:
            scaler = StandardScaler()
        # Fit the scaler to the labels
            scaler.fit_transform(labels.reshape(-1, 1))

        scalers.append(scaler)


    if not os.path.exists(gpr_file):
            raise FileNotFoundError(f"Checkpoint not found at {gpr_file}")

    # now load the normalized features and labels up to that point
    if os.path.exists(norm_features_file):
        features_norm = torch.tensor(pd.read_csv(norm_features_file).values).float()
    else:
        raise FileNotFoundError(f"Normalized features file not found at {norm_features_file}")

    if os.path.exists(norm_labels_file):
        labels_norm = torch.tensor(pd.read_csv(norm_labels_file).values).float()
    else:
        raise FileNotFoundError(f"Normalized labels file not found at {norm_labels_file}")
    # Check if features and labels have the same number of samples
    if features_norm.size(0) != labels_norm.size(0):
        raise ValueError(f"Features and labels must have the same number of samples. Found {features_norm.size(0)} features and {labels_norm.size(0)} labels.")


    # Load the saved state dictionary from file
    checkpoint = torch.load(gpr_file, weights_only=True)

        # Instantiate the likelihood and model classes
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    model = MultitaskGPRegressionModel(features_norm, labels_norm, likelihood, num_tasks=2)

        # Load the model and optimizer state dicts
    model.load_state_dict(checkpoint['model'])

    return model, likelihood, scalers

def standard_normalize_features(reshaped_features, normalization_stats):
    std_normal_dict = normalization_stats['std_normal_dict']
    min_L = normalization_stats['min_L']
    max_L = normalization_stats['max_L']
    maxS = normalization_stats['maxS']

    # Apply standard normalization based on the loaded stats (same as before)
    reshaped_features[:20] /= reshaped_features[20]
    reshaped_features[23] /= reshaped_features[20]
    reshaped_features[24] /= reshaped_features[20]
    reshaped_features[25] /= reshaped_features[20]
    reshaped_features[26] /= reshaped_features[20]
    reshaped_features[28] /= reshaped_features[20]

    reshaped_features[20] = (reshaped_features[20] - min_L) / (max_L - min_L)

    std_norm_indices = {
        'SCD': 21, 'SHD': 22, '|net charge|': 23, 'sum lambda': 24,
        'beads(+)': 25, 'beads(-)': 26, 'shannon_entropy': 27, 'mol wt': 28
    }

    for feat, idx in std_norm_indices.items():
        if feat == 'shannon_entropy':
            reshaped_features[idx] = reshaped_features[idx] / maxS - 1
        else:
            mean_val, std_val = std_normal_dict[feat]
            reshaped_features[idx] = (reshaped_features[idx] - mean_val) / std_val

    return reshaped_features

def load_normalization_stats(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def generate_umap_visualizations(feats_total, children_folder, featurizer, normalization_stats, model, likelihood, ehvi='standard', exploration='similarity_penalty', transform='log'):
    """
    Loads all child sequences from the folder, featurizes and normalizes them,
    then saves features to a CSV and generates UMAP visualizations.
    """

    # Get all child sequence files
    child_files = sorted(glob(os.path.join(children_folder, 'seq_child_*.txt')))
    #print(f"here are the child files {child_files}", flush=True)
    seqs = []
    for file in child_files:
        with open(file, 'r') as f:
            seq = f.readline().strip()
            seqs.append(seq)

    # Featurize and normalize
    features = []
    for seq in seqs:
        #print(seq, flush=True)
        raw_feats = featurizer.featurize(seq)
        norm_feats = standard_normalize_features(np.asarray(raw_feats), normalization_stats)
        features.append(norm_feats)
        #print(norm_feats, flush=True)

    features = np.array(features)

    feat_values = feats_total.values
    features_total = []

    for i in range(feat_values.shape[0]):
        features_total.append(standard_normalize_features(feat_values[i,:], normalization_stats))

    features_total = np.array(features_total)
    reducer_full = UMAP(random_state=42)
    #print(features.shape)
    reducer_full.fit(features_total)
    embedding_full = reducer_full.transform(features)
    df_feats = pd.DataFrame(features)
    df_feats.to_csv(os.path.join(children_folder, 'child_features.csv'), index=False)

    # Predict B2 and Diff
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = likelihood(model(torch.tensor(features).float())).mean.detach().numpy()

    b2_pred = preds[:, 0]
    diff_pred = preds[:, 1]

    style = plotting.PlotStyle(
        font_family="Arial",
        base_font_size=14,
        label_font_size=16,
        title_font_size=16,
        tick_font_size=14,
        legend_font_size=16,
        max_xticks=20,
        max_yticks=20,
        tight_layout=True,
        transparent=False,
    )
    plotting.set_plot_style(style)
    FIGSIZE = (6, 5)
    FMT = 'png'
    DPI = 450

    # --- Full UMAP --- #

    fig, ax = plt.subplots(figsize=(4, 5))
    sc = ax.scatter(x=embedding_full[:, 0], y=embedding_full[:, 1], c=diff_pred, cmap='viridis', s=50)
    ax.set_title("UMAP: Full Features Colored by Diffusivity")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    plt.colorbar(sc, label="Diffusivity")
    plt.tight_layout()
    plotting.format_and_save_figure(
                fig=fig,
                axes=ax,
                save_path=f"umap_full_diff_{ehvi}_{exploration}_{transform}.png",
                dpi=DPI,
                fmt=FMT,
                dimensions=FIGSIZE,
                style=style,
                save_colorbar=True,
                colorbar_label=True,  # your plotting.py uses True to keep existing label
                close=True,
                label=False,
            )
    #fig.savefig(os.path.join(children_folder, f"umap_full_diff_{ehvi}_{exploration}_{transform}.png"), bbox_inches='tight', dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(4, 5))
    sc = ax.scatter(x=embedding_full[:, 0], y=embedding_full[:, 1], c=b2_pred, cmap='viridis', s=50)
    ax.set_title("UMAP: Full Features Colored by exp density")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    plt.colorbar(sc, label="Exp_Density")
    plt.tight_layout()
    plotting.format_and_save_figure(
                fig=fig,
                axes=ax,
                save_path=f"umap_full_expdens_{ehvi}_{exploration}_{transform}.png",
                dpi=DPI,
                fmt=FMT,
                dimensions=FIGSIZE,
                style=style,
                save_colorbar=True,
                colorbar_label=True,  # your plotting.py uses True to keep existing label
                close=True,
                label=False,
            )
    #fig.savefig(os.path.join(children_folder, f"umap_full_expdens_{ehvi}_{exploration}_{transform}.png"), bbox_inches='tight', dpi=300)
    plt.close()

    # --- Physicochemical UMAP --- #
    physchem_features_total = features_total[:, 20:]
    physchem_features = features[:,20:]
    reducer_phys = UMAP(random_state=42)
    reducer_phys.fit(physchem_features_total)

    embedding_phys = reducer_phys.transform(physchem_features)
    fig, ax = plt.subplots(figsize=(4, 4))
    sc = ax.scatter(x=embedding_phys[:, 0], y=embedding_phys[:, 1], c=b2_pred, cmap='magma', s=50)
    ax.set_title("UMAP: Physicochemical Features Colored by exp_density")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax = plt.gca()
    #ax.tick_params(axis='x', labelsize=20)
    #ax.tick_params(axis='y', labelsize=20)
    plt.colorbar(sc, label="exp_density")
    plt.tight_layout()
    plotting.format_and_save_figure(
                fig=fig,
                axes=ax,
                save_path=f"umap_physchem_expdens_{ehvi}_{exploration}_{transform}.png",
                dpi=DPI,
                fmt=FMT,
                dimensions=FIGSIZE,
                style=style,
                save_colorbar=True,
                colorbar_label=True,  # your plotting.py uses True to keep existing label
                close=True,
                label=False,
            )
    #fig.savefig(os.path.join(children_folder, f"umap_physchem_expdens_{ehvi}_{exploration}_{transform}.png"), bbox_inches='tight', dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(4, 4))
    sc = ax.scatter(x=embedding_phys[:, 0], y=embedding_phys[:, 1], c=diff_pred, cmap='magma', s=50)
    ax.set_title("UMAP: Physicochemical Features Colored by diffusivity")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    #ax.tick_params(axis='x', labelsize=20)
    #ax.tick_params(axis='y', labelsize=20)
    plt.colorbar(sc, label="diffusivity")
    plt.tight_layout()
    plotting.format_and_save_figure(
                fig=fig,
                axes=ax,
                save_path=f"umap_physchem_diff_{ehvi}_{exploration}_{transform}.png",
                dpi=DPI,
                fmt=FMT,
                dimensions=FIGSIZE,
                style=style,
                save_colorbar=True,
                colorbar_label=True,  # your plotting.py uses True to keep existing label
                close=True,
                label=False,
            )
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Run Genetic Algorithm for intermediate iterations of active learning.')
    parser.add_argument('--gen_folder', type=str, required=True, help='Folder where the genetic algorithm produces children.')
    parser.add_argument('--iter_folder', type=str, required=True, help='Folder where current generation is located, eg features, labels.')
    parser.add_argument('--iteration', type=int, required=True, help='Iteration number for the genetic algorithm.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the ML models (e.g. GPR and RF models).')
    parser.add_argument('--obj1', type=str, required=True, help='Objective 1 for the Pareto front.')
    parser.add_argument('--obj2', type=str, required=True, help='Objective 2 for the Pareto front.')
    parser.add_argument('--ehvi_variant', type=str, choices=['standard', 'epsilon'], default='epsilon',
                    help='Type of EHVI strategy: standard or epsilon.')
    parser.add_argument('--exploration_strategy', type=str, choices=['similarity_penalty', 'kriging_believer', 'constant_liar_min', 'constant_liar_mean', 'constant_liar_max', 'standard'], default='similarity_penalty',
                    help='Exploration method: similarity_penalty or front_augmentation.')
    parser.add_argument('--transform', type=str, choices=['yeoj', 'log'], default='log',
                    help='Transformation applied to the labels: yeojohnson or log.')
    parser.add_argument('--monte_carlo', type=str, required=False, default=None, help='Whether to use Monte Carlo sampling for EHVI calculation. If provided, it will be used as a flag.')
    parser.add_argument('--acquisition_test', action='store_true', help='Whether to run acquisition test UMAP generation')
    parser.add_argument('--front', type=str, required=True, help='Front type: upper or lower.')
    args = parser.parse_args()
    db_path = "/home/zl4808/scripts/GENDATA/databases"
    model_name = args.gen_folder.split('/')[6]
    assert model_name in ['HPS_URRY', 'MPIPI', 'HPS_KR', 'MARTINI', 'CALVADOS'], "Model name not recognized. Please check the model name."
    # Initialize the sequence featurizer
    featurizer = sf.SequenceFeaturizer(model_name.lower(), db_path)
    #iteration_folder = args.curr_folder + f'/iteration_{args.iteration}'

    # feats_total = pd.read_csv(os.path.join(args.iter_folder, f"features_gen{args.iteration}_TEMP.csv")) if args.exploration_strategy == 'front_augmentation' \
    #     else pd.read_csv(os.path.join(args.iter_folder, f"features_gen{args.iteration}.csv"))

    if args.acquisition_test:
        feats_total = pd.read_csv(f"/scratch/gpfs/zl4808/PROJECTS/ACQUISITION_COMPARISONS/{model_name}/GENERATIONS/iteration_{args.iteration}/features_gen{args.iteration}.csv")
    else:
        feats_total = pd.read_csv(f"/scratch/gpfs/zl4808/PROJECTS/MODEL_COMPARISON/{model_name}/GENERATIONS/iteration_{args.iteration}/features_gen{args.iteration}.csv")
    objectives = [args.obj1, args.obj2]
    print(f"objectives:{objectives}", flush=True)

    if args.monte_carlo is not None:
        transform = f"{args.transform}_MC"
    else:
        transform = args.transform
        #print(f"Using Monte Carlo sampling for EHVI calculation with transform: {args.transform}", flush=True)

    normalization_stats = load_normalization_stats(os.path.join(args.iter_folder, f'normalization_stats.json'))
    model, likelihood, scalers = load_gpr_models(args.model_path, args.iter_folder, args.iteration, objectives, args.ehvi_variant, args.exploration_strategy, 2, transform, args.front)
    print("models and stats loaded", flush=True)
    model.eval()
    likelihood.eval()

    children_folder = os.path.join(args.gen_folder, f"children_{args.ehvi_variant}_{args.exploration_strategy}_{transform}_{args.front}")
    print(children_folder)
    generate_umap_visualizations(feats_total, children_folder, featurizer, normalization_stats, model, likelihood, ehvi=args.ehvi_variant, exploration=args.exploration_strategy, transform=transform)

if __name__ == '__main__':
        main()
