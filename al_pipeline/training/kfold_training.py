from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
import gpytorch
import pandas as pd
import json
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import r2_score

from al_pipeline.core.config import ALConfig
from al_pipeline.data_prep.data_loading import load_dataset, convert_and_normalize_features

from .ml_models import GPRegressionModel, MultitaskGPRegressionModel
from .trainers import GPRTrainer, MultitaskGPRTrainer


def train_from_config(cfg: ALConfig, log=None) -> None:
    """
    Train the AL surrogate model based on the configuration.

    Parameters:
    -----------
    cfg : ALConfig
        Configuration that specifies paths, training params, etc. See al_pipeline.core.config.py for details.
    log : Logger, optional
        If provided, used for logging.

    Returns:
    -----------
    None
    """

    train_model_type = cfg.train_model_type

    if train_model_type == "gpr_multitask":
        _kfold_gpr_multitask_from_config(cfg, log=log)
    elif train_model_type == "gpr_singletask":
        for label_col in [cfg.obj1, cfg.obj2]:
            _kfold_gpr_single_from_config(cfg, label_column=label_col, log=log)
    elif train_model_type == "dnn":
        _kfold_dnn_from_config(cfg, log=log) # TODO: implement this function
    else:
        raise ValueError(f"Unknown train_model_type: {train_model_type}")

def _kfold_gpr_multitask_from_config(cfg: ALConfig, log=None):
    """
    Multitask GPR CV for the AL surrogate model.

    Parameters: 
    -----------
    cfg : ALConfig
        Configuration that specifies paths, training params, etc. See al_pipeline.core.config.py for details.
    log : logger, Optional
        Logger object or None. If provided, used for logging.

    Returns:
    -----------
    None
    """

    if log:
        log.info("Starting k-fold training for multitask GPR...")
        
    features_file    = cfg.paths.features_csv
    labels_file      = cfg.paths.labels_csv
    feats_norm_file  = cfg.paths.features_norm_csv
    labels_norm_file = cfg.paths.labels_norm_csv
    norm_stats_file  = cfg.paths.norm_stats
    label_columns    = [cfg.obj1, cfg.obj2]
    k                = cfg.k_folds
    epochs           = cfg.epochs
    patience         = cfg.patience
    save_best_fold   = cfg.save_best_fold
    lr               = cfg.learning_rate
    transform        = cfg.transform
    exploration      = cfg.exploration_strategy
    ehvi_variant     = cfg.ehvi_variant

    if log:
        log.info(f"Training params: \n"
                f"kfold = {k} \n"
                f"epochs={epochs}, \n"
                f"patience={patience}, \n"
                f"lr={lr}, \n"
                f"transform={transform},\n"
                f"exploration={exploration}, \n"
                f"ehvi_variant={ehvi_variant}")

    feats_raw, labels = load_dataset(
        features_file, labels_file, label_columns=label_columns
    )  # feats_raw: (N,29), labels_raw: (N,2)


    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    train_mses, test_mses = [], []
    model_dicts, likelihood_dicts = [], []

    fold = 0
    best_val_mse = float("inf")

    for train_idx, val_idx in kf.split(feats_raw):
        fold += 1
        print(f"Fold {fold}")

        train_feats_raw = feats_raw[train_idx]
        test_feats_raw  = feats_raw[val_idx]
        train_labels    = labels[train_idx].copy()
        test_labels     = labels[val_idx].copy()

        # normalize features per fold
        train_feats_norm, stats = convert_and_normalize_features(train_feats_raw, train=True)
        test_feats_norm         = convert_and_normalize_features(test_feats_raw, train=False, stats=stats)

        train_feats = torch.tensor(train_feats_norm).float()
        test_feats  = torch.tensor(test_feats_norm).float()

        # label scaling per train/val split
        

        for i, label_col in enumerate(label_columns):
            if transform == "log":
                if np.any(train_labels[:, i] <= 0):
                    raise ValueError(f"Log transform requires all positive values, but found non-positive in column {label_col}")
                #if label_col == "diff": #TODO: implement log transform for diffusivity
                    
                scaler = StandardScaler()
            elif transform == "yeoj":
                scaler = PowerTransformer(method='yeo-johnson', standardize=True)
            else:
                raise ValueError(f"Unknown transform: {transform}")
            train_labels[:, i] = scaler.fit_transform(train_labels[:, i].reshape(-1, 1)).flatten()
            test_labels[:, i] = scaler.transform(test_labels[:, i].reshape(-1, 1)).flatten()
        
        train_labels = torch.tensor(train_labels).float()
        test_labels  = torch.tensor(test_labels).float()

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
        model = MultitaskGPRegressionModel(train_feats, train_labels, likelihood, num_tasks=2)

        trainer = MultitaskGPRTrainer(model, likelihood, learning_rate=lr, epochs=epochs, patience=patience)
        train_log = trainer.train((train_feats, train_labels), (test_feats, test_labels), early_stop=True)

        train_mse = trainer.evaluate(train_feats, train_labels)
        test_mse  = trainer.evaluate(test_feats, test_labels)
        train_mses.append(train_mse)
        test_mses.append(test_mse)

        if log:
            log.info(f"Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")

        model.eval(); likelihood.eval()
        model_dicts.append(model.state_dict())
        likelihood_dicts.append(likelihood.state_dict())

        if save_best_fold and test_mse < best_val_mse:
            best_val_mse = test_mse
            best_path = cfg.paths.models_dir / f"GPR_multitask_iter{cfg.iteration}_{cfg.paths.tag}_BEST_FOLD.pt"
            save_chkpt(best_path, model)

    if log:
        log.info(
            f"Final multitask k-fold: Train MSE {np.mean(train_mses):.4f}+-{np.std(train_mses):.4f}, "
            f"Test MSE {np.mean(test_mses):.4f}+-{np.std(test_mses):.4f}"
        )

    # After averaging hyperparameters across folds
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)

    # scale labels
    for i, label_col in enumerate(label_columns):
        if transform == "log":
            if np.any(labels[:, i] <= 0):
                raise ValueError(f"Log transform requires all positive values, but found non-positive in column {label_col}")
            scaler = StandardScaler()
        elif transform == "yeoj":
            scaler = PowerTransformer(method='yeo-johnson', standardize=True)
        else:
            raise ValueError(f"Unknown transform: {transform}")
        labels[:, i] = scaler.fit_transform(labels[:, i].reshape(-1, 1)).flatten()
        
    # scale all labels
    labels = torch.tensor(labels).float()

    # scale all features
    feats, norm_stats = convert_and_normalize_features(feats_raw, train=True)
    feats = torch.tensor(feats).float()

    # save normalization stats
    with open(norm_stats_file, "w") as f:
        json.dump(norm_stats, f)

    model = MultitaskGPRegressionModel(feats, labels, likelihood, num_tasks = 2)

    # average hyperparameters
    avg_model_state = {}
    for key in model_dicts[0].keys():
        avg_model_state[key] = sum(d[key] for d in model_dicts) / len(model_dicts)

    avg_likelihood_state = {}
    for key in likelihood_dicts[0].keys():
        avg_likelihood_state[key] = sum(d[key] for d in likelihood_dicts) / len(likelihood_dicts)
    
    
    model.load_state_dict(avg_model_state)
    likelihood.load_state_dict(avg_likelihood_state)

    # Train the final model on the entire dataset
    final_trainer = MultitaskGPRTrainer(model, likelihood, learning_rate=lr, epochs=epochs, patience=patience)
    final_log = final_trainer.train((feats, labels), None, early_stop=False)

    model.eval()
    likelihood.eval()

    with torch.no_grad():
        pred  = likelihood(model(feats)).mean.detach().numpy()
    
    labels = labels.detach().numpy()

    # next, save normalized features/labels used to train the multitask GPR
    feats_orig_df = pd.read_csv(features_file)

    features_train_df = pd.DataFrame(feats.detach().numpy(), columns=feats_orig_df.columns)
    labels_train_df   = pd.DataFrame(labels, columns=label_columns)

    features_train_df.to_csv(feats_norm_file, index=False)
    labels_train_df.to_csv(labels_norm_file, index=False)

    if log:
        log.info(f"Saved normalized features and labels for {cfg.iteration}, front {cfg.front}.")

    # plot the parity plots for each task
    for i, label in enumerate(label_columns):
        fig, ax = plt.subplots(figsize=(2, 2), dpi=300)

        # Extract true and predicted values for task `i`
        y = labels[:, i]
        y_pred = pred[:, i]


        r2 = r2_score(y, y_pred)

        ax.scatter(y, y_pred, color='orange', alpha=0.3, label=f"$R^2$={r2:.4f}")
        #ax.scatter(y_test, y_test_pred, color='orange', alpha=0.3)

        ax.set_xlabel(f'True {label}', fontsize=6)
        ax.set_ylabel(f'Predicted {label}', fontsize=6)
        ax.set_title(f'GPR Performance — {label}', fontsize=6)


        ax.plot([min(y), max(y)], [min(y), max(y)], 'r--')

        ax.tick_params(axis='both', which='both', labelsize=4, direction='in')
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig_path = (cfg.paths.models_dir / f"GPR_multitask_iter{cfg.iteration}_{cfg.paths.tag}_FIT_{label}.png")
        fig.savefig(str(fig_path), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save final model")
    save_chkpt(cfg.paths.gpr_multitask_chkpt(temp=False), model)

    if cfg.exploration_strategy in ['standard', 'similarity_penalty']:
        save_chkpt(cfg.paths.gpr_multitask_chkpt(temp=True), model)

    print("Final model saved with training on the entire dataset")


def _kfold_gpr_single_from_config(cfg: ALConfig, label_column: str, log=None) -> None:
    """
    Single-task GPR training mirroring _kfold_gpr_multitask_from_config conventions.

    Parameters:
    -----------
    cfg : ALConfig
        Configuration that specifies paths, training params, etc. See al_pipeline.core.config.py for details.
    label_column : str      
        The name of the label column to train on (e.g., exp_denisty, diff, or another label).
    log : Logger object or None. If provided, used for logging.

    Returns:
    -----------
    None
    """

    if log:
        log.info(f"Starting k-fold training for single-task GPR ({label_column})...")

    features_file   = cfg.paths.features_csv
    labels_file     = cfg.paths.labels_csv

    # normalized outputs
    feats_norm_file  = cfg.paths.features_norm_csv
    # for the single task gprs, we have a label file per objective
    labels_norm_file = labels_file.with_stem(labels_file.stem + f"_{label_column}_NORM_{cfg.paths.tag}")
    norm_stats_file  = cfg.paths.norm_stats

    k               = cfg.k_folds
    epochs          = cfg.epochs
    patience        = cfg.patience
    save_best_fold  = cfg.save_best_fold
    lr              = cfg.learning_rate
    transform       = cfg.transform
    exploration     = cfg.exploration_strategy
    ehvi_variant    = cfg.ehvi_variant

    if log:
        log.info(
            f"Training params: \n"
            f"kfold = {k} \n"
            f"epochs={epochs}, \n"
            f"patience={patience}, \n"
            f"lr={lr}, \n"
            f"transform={transform},\n"
            f"exploration={exploration}, \n"
            f"ehvi_variant={ehvi_variant}"
        )

    # load the data
    feats_raw, labels_raw = load_dataset(
        str(features_file), str(labels_file), label_columns=label_column
    )  # feats_raw: (N,F), labels_raw: (N,1) or (N,)

    # force (N,1) float array for consistent naming with multitask
    labels = np.asarray(labels_raw, dtype=float)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    # K-Fold setup 
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    train_mses, test_mses = [], []
    model_dicts, likelihood_dicts = [], []

    # for single-task, we also collect kernel/noise hyperparams
    # and initialize the final retrain with their averages
    length_scales, output_scales, noise_vars = [], [], []

    fold = 0
    best_val_mse = float("inf")

    for train_idx, val_idx in kf.split(feats_raw):
        fold += 1
        print(f"Fold {fold}")

        train_feats_raw = feats_raw[train_idx]
        test_feats_raw  = feats_raw[val_idx]
        train_labels    = labels[train_idx].copy()   # (n_tr,1)
        test_labels     = labels[val_idx].copy()     # (n_te,1)

        # normalize features per fold
        train_feats_norm, stats = convert_and_normalize_features(train_feats_raw, train=True)
        test_feats_norm         = convert_and_normalize_features(test_feats_raw, train=False, stats=stats)

        train_feats = torch.tensor(train_feats_norm).float()
        test_feats  = torch.tensor(test_feats_norm).float()

        # label scaling per train/val split (single task => i=0)
        if transform == "log":
            if np.any(train_labels[:, 0] <= 0):
                raise ValueError(
                    f"Log transform requires all positive values, but found non-positive in {label_column}"
                )
            # TODO: this implementation of 'log' isn't really right, we use it prior to standard scaling in case one 
            # of the objectives has a much larger scale than the other (not always the best idea but worth including)
            train_tf = np.log(train_labels + 1e-8).reshape(-1, 1)
            test_tf  = np.log(test_labels + 1e-8).reshape(-1, 1)
            scaler = StandardScaler()

        elif transform == "yeoj":
            train_tf = train_labels.reshape(-1, 1)
            test_tf  = test_labels.reshape(-1, 1)
            scaler = PowerTransformer(method="yeo-johnson", standardize=True)

        else:
            raise ValueError(f"Unknown transform: {transform}")

        train_labels[:, 0] = scaler.fit_transform(train_tf).flatten()
        test_labels[:, 0]  = scaler.transform(test_tf).flatten()

        train_labels_t = torch.tensor(train_labels).float().flatten()  # (n_tr,)
        test_labels_t  = torch.tensor(test_labels).float().flatten()   # (n_te,)

        # build single-task GP
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GPRegressionModel(train_feats, train_labels_t, likelihood, kernel=None)

        trainer = GPRTrainer(model, likelihood, learning_rate=lr, epochs=epochs, patience=patience)
        _train_log = trainer.train((train_feats, train_labels_t), (test_feats, test_labels_t), early_stop=True)

        train_mse = trainer.evaluate(train_feats, train_labels_t)
        test_mse  = trainer.evaluate(test_feats, test_labels_t)
        train_mses.append(train_mse)
        test_mses.append(test_mse)

        if log:
            log.info(f"[{label_column}] Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")

        model.eval(); likelihood.eval()
        model_dicts.append(model.state_dict())
        likelihood_dicts.append(likelihood.state_dict())

        # capture hyperparams for retrain initialization
        # (handle scalar vs batched lengthscale robustly)
        ls = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
        length_scales.append(float(np.mean(ls)))
        output_scales.append(float(model.covar_module.outputscale.detach().cpu().item()))
        noise_vars.append(float(likelihood.noise.detach().cpu().item()))

        if save_best_fold and test_mse < best_val_mse:
            best_val_mse = test_mse
            best_path = (
                cfg.paths.models_dir
                / f"GPR_iter{cfg.iteration}_{label_column}_{cfg.paths.tag}_BEST_FOLD.pt"
            )
            save_chkpt(best_path, model)

    if log:
        log.info(
            f"Final single-task k-fold ({label_column}): "
            f"Train MSE {np.mean(train_mses):.4f}+-{np.std(train_mses):.4f}, "
            f"Test MSE {np.mean(test_mses):.4f}+-{np.std(test_mses):.4f}"
        )

    # Retrain final model on full dataset with averaged hyperparams
    # normalize all features
    feats_norm, norm_stats = convert_and_normalize_features(feats_raw, train=True)
    feats = torch.tensor(feats_norm).float()

    # Save normalization stats
    with open(norm_stats_file, "w") as f:
        json.dump(norm_stats, f)

    # transform + scale labels on full dataset
    labels_full = labels.copy()  # (N,1)

    if transform == "log":
        if np.any(labels_full <= 0):
            raise ValueError(
                f"Log transform requires all positive values, but found non-positive in {label_column}"
            )
        full_tf = np.log(labels_full + 1e-8).reshape(-1, 1)
        full_scaler = StandardScaler()

    elif transform == "yeoj":
        full_tf = labels_full.reshape(-1, 1)
        full_scaler = PowerTransformer(method="yeo-johnson", standardize=True)

    else:
        raise ValueError(f"Unknown transform: {transform}")

    labels_full = full_scaler.fit_transform(full_tf).flatten()
    labels_t = torch.tensor(labels_full).float().flatten()  # (N,)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPRegressionModel(feats, labels_t, likelihood, kernel=None)

    # initialize hyperparams from fold averages (stabilizes retrain)
    model.covar_module.base_kernel.lengthscale = torch.tensor(np.mean(length_scales))
    model.covar_module.outputscale = torch.tensor(np.mean(output_scales))
    likelihood.noise = torch.tensor(np.mean(noise_vars))

    final_trainer = GPRTrainer(model, likelihood, learning_rate=lr, epochs=epochs, patience=patience)
    _final_log = final_trainer.train((feats, labels_t), None, early_stop=False)

    model.eval()
    likelihood.eval()

    # Save normalized training data (single-task version)
    feats_orig_df = pd.read_csv(str(features_file))
    features_train_df = pd.DataFrame(feats.detach().cpu().numpy(), columns=feats_orig_df.columns)
    labels_train_df   = pd.DataFrame(labels_full, columns=[label_column])

    features_train_df.to_csv(str(feats_norm_file), index=False)
    labels_train_df.to_csv(str(labels_norm_file), index=False)

    if log:
        log.info(f"Saved normalized features: {feats_norm_file}")
        log.info(f"Saved normalized labels ({label_column}): {labels_norm_file}")

    # parity plot for final model on full dataset
    with torch.no_grad():
        pred = likelihood(model(feats)).mean.detach().cpu().numpy().flatten()

    y = labels_full.flatten()
    y_pred = pred 
    r2 = r2_score(y, y_pred)

    fig, ax = plt.subplots(figsize=(2, 2), dpi=300)
    ax.scatter(y, y_pred, color="orange", alpha=0.3, label=f"$R^2$={r2:.4f}")
    ax.plot([min(y), max(y)], [min(y), max(y)], "r--")
    ax.set_xlabel(f"True {label_column}", fontsize=6)
    ax.set_ylabel(f"Predicted {label_column}", fontsize=6)
    ax.set_title(f"GPR Performance — {label_column}", fontsize=6)
    ax.tick_params(axis="both", which="both", labelsize=4, direction="in")
    ax.legend(fontsize=6)
    fig.tight_layout()

    fig_path = (
        cfg.paths.models_dir
        / f"GPR_iter{cfg.iteration}_{label_column}_{cfg.paths.tag}_FIT.png"
    )
    fig.savefig(str(fig_path), dpi=300, bbox_inches="tight")
    plt.close()

    # Save final model
    save_chkpt(cfg.paths.gpr_singletask_chkpt([label_column], temp=False)[0], model)

    if cfg.exploration_strategy in ["standard", "similarity_penalty"]:
        save_chkpt(cfg.paths.gpr_singletask_chkpt([label_column], temp=True)[0], model)

    print("Final model saved with training on the entire dataset")


#### TODO: implement DNN path ####
# def _dnn_from_config(cfg: ALConfig, log=None):



def save_chkpt(model_path: str | Path, model, optimizer=None, val_losses=None, train_losses=None, trained=False) -> None:
    """
    Save a training checkpoint.

    Parameters:
    -----------
        model_path: str | Path 
            The path to save the model to.
        model: torch.nn.Module
            The model to save.
        optimizer: torch.optim.Optimizer
            The optimizer to save (optional).
        val_losses: list of float
            A list containing the validation losses.
        train_losses: list of float
            A list containing the training losses.
    """
    if trained==True:
        state_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
        }
    else:
        state_dict = {
            'model': model.state_dict()
        }
    torch.save(state_dict, model_path)


#### Utility function to average model weights (not used currently) ####
def average_model_weights(models):
    """
    Average the weights of the models across K folds.
    
    Parameters:
    -----------
    models: List
      List of model state_dicts.
    
    Returns:
    --------
    avg_state_dict: dict
        Averaged state_dict for the model.
    """
    num_models = len(models)
    avg_state_dict = models[0].copy()  # Initialize with the state_dict from the first model
    for key in avg_state_dict:
        # Average the values across models
        avg_state_dict[key] = torch.stack([models[i][key] for i in range(num_models)], dim=0).mean(dim=0)
    return avg_state_dict



    