from __future__ import annotations

from al_pipeline.core.config import ALConfig
from al_pipeline.training.ml_models import MultitaskGPRegressionModel, GPRegressionModel
from al_pipeline.training.kfold_training import save_chkpt
from al_pipeline.data_prep.data_loading import load_dataset, convert_and_normalize_features
from .ga_utils import load_normalization_stats, load_models
import al_pipeline.featurization.sequence_featurizer as sf
import numpy as np
import torch
import pandas as pd
import gpytorch


def _retrain_model_gpr_singletask(cfg: ALConfig, model, likelihood, train_X, train_y):
    # Create a new GPR model and likelihood
    likelihood_new = gpytorch.likelihoods.GaussianLikelihood()
    model_new      = GPRegressionModel(train_X, train_y, likelihood_new)

    likelihood_new.load_state_dict(likelihood.state_dict())
    model_new.load_state_dict(model.state_dict())

    # Set the model and likelihood to training mode
    model_new.train()
    likelihood_new.train()
    # Set the model to use the same device as the data
    device = train_X.device if train_X.is_cuda else 'cpu'
    model_new.to(device)
    likelihood_new.to(device)

    # Define the optimizer
    optimizer = torch.optim.Adam(model_new.parameters(), lr=cfg.learning_rate/100)  # Use a lower learning rate for fine-tuning
    # Define the loss function
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_new, model_new)
    # Training loop
    num_epochs = 1  # Adjust the number of epochs as needed
    for epoch in range(num_epochs):

        optimizer.zero_grad()  # Zero gradients from previous iteration
        output = model_new(train_X)  # Forward pass
        loss = -mll(output, train_y)  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update the model parameters
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}", flush=True)
    model_new.eval()
    likelihood_new.eval()
    return model_new, likelihood_new

def _retrain_model_gpr_multitask(cfg: ALConfig, model, likelihood, train_X, train_y):
    # Create a new GPR model and likelihood
    likelihood_new = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    model_new      = MultitaskGPRegressionModel(train_X, train_y, likelihood_new, num_tasks=2)

    likelihood_new.load_state_dict(likelihood.state_dict())
    model_new.load_state_dict(model.state_dict())

    # Set the model and likelihood to training mode
    model_new.train()
    likelihood_new.train()

    # Set the model to use the same device as the data
    device = train_X.device if train_X.is_cuda else 'cpu'
    model_new.to(device)
    likelihood_new.to(device)
    # Define the optimizer
    optimizer = torch.optim.Adam(model_new.parameters(), lr=cfg.learning_rate/100)  # Use a lower learning rate for fine-tuning
    # Define the loss function
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_new, model_new)
    # Training loop
    num_epochs = 1  # Adjust the number of epochs as needed
    for epoch in range(num_epochs):
        optimizer.zero_grad()  # Zero gradients from previous iteration
        output = model_new(train_X)  # Forward pass
        loss = -mll(output, train_y)  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update the model parameters
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}", flush=True)
    model_new.eval()
    likelihood_new.eval()
    return model_new, likelihood_new

def retrain_model(cfg: ALConfig, model_bundle, train_X, train_y):
    if cfg.train_model_type == "gpr_multitask":
        model_old, likelihood_old = model_bundle["model"], model_bundle["likelihood"]
        model_new, likelihood_new = _retrain_model_gpr_multitask(cfg, model_old, likelihood_old, train_X, train_y)
    elif cfg.train_model_type == "gpr_singletask":
        models      = []
        likelihoods = []
        models_old      = model_bundle["models"]
        likelihoods_old = model_bundle["likelihoods"]
        objectives = [cfg.obj1, cfg.obj2]
        for component, label in enumerate(objectives):
            model_new, likelihood_new = _retrain_model_gpr_singletask(cfg, models_old[label], likelihoods_old[label], train_X, train_y[:, component].flatten())
            models.append(model_new)
            likelihoods.append(likelihood_new)
        model_new = models
        likelihood_new = likelihoods
    else:
        raise ValueError(f"Unknown model type: {cfg.train_model_type} not implemented yet.")
    return model_new, likelihood_new

def predict_for_augmentation(model_bundle, Xn, cfg, return_std: bool = False):
    if cfg.train_model_type == "gpr_multitask":
        model = model_bundle["model"]
        Xn = torch.tensor(Xn.copy(), dtype=torch.float32)
        if Xn.ndim == 1:
            Xn = Xn.view(1,-1)
        mu, cov = get_cand_stats(model, Xn)
        # mu: (N,2), cov: (N,2,2)

    elif cfg.train_model_type == "gpr_singletask":
        # No cross-objective covariance for single-task GPs — diagonal cov.
        # Goes direct to the per-objective models rather than through the
        # surrogate ABC: the surrogate's `predict_pool` takes raw features
        # (so MoE/global can share the contract), but here we already have
        # normalized features handy and don't need MoE in the augmentation
        # path. Direct calls are simpler than re-inverting the normalization.
        Xn = torch.tensor(Xn.copy(), dtype=torch.float32)
        if Xn.ndim == 1:
            Xn = Xn.view(1,-1)
        models = model_bundle["models"]
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            post1 = models[cfg.obj1](Xn)
            post2 = models[cfg.obj2](Xn)
            mu = np.column_stack([
                post1.mean.detach().cpu().numpy().reshape(-1),
                post2.mean.detach().cpu().numpy().reshape(-1),
            ])
            std = np.column_stack([
                post1.stddev.detach().cpu().numpy().reshape(-1),
                post2.stddev.detach().cpu().numpy().reshape(-1),
            ])
        cov = np.zeros((mu.shape[0], 2, 2), dtype=np.float32)
        cov[:,0,0] = std[:,0]**2
        cov[:,1,1] = std[:,1]**2
        
    else:
        raise ValueError(f"Unknown model type: {cfg.train_model_type} not implemented yet.")
    
    if return_std:
        std = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))  # (N,2)
        return mu, cov, std
    return mu, cov


def get_cand_stats(model, features):
    features = torch.tensor(features).float() if not torch.is_tensor(features) else features
    features = features.reshape(1,-1) if features.ndim == 1 else features

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = model(features)
        mus = posterior.mean
        cov = posterior.covariance_matrix # has shape (Ntasks * Nsamples, Ntasks * Nsamples)
        cov = cov.reshape(mus.size()[0], cov.size()[0]//mus.size()[0], mus.size()[0], cov.size()[0]//mus.size()[0]) # (Nsamples, Ntasks, Nsamples, Ntasks) --> Cov for sample i is cov[i, :, i, :] 
        covs = cov[torch.arange(mus.shape[0]), :, torch.arange(mus.shape[0]), :] # (Nsamples, Ntasks, Ntasks) because we are getting the "diagonal" of the larger covariance matrix

    return mus.detach().numpy(), covs.detach().numpy()



def overlap_cs_joint(mu, A, nu, B):
    """
    Compute the overlap between two multivariate Gaussian distributions with means mu and nu, and covariance matrices A and B.
    Parameters:
    -----------
    mu: np.ndarray
        Mean vector of the first Gaussian distribution.
    A: np.ndarray
        Covariance matrix of the first Gaussian distribution.
    nu: np.ndarray
        Mean vector of the second Gaussian distribution.
    B: np.ndarray
        Covariance matrix of the second Gaussian distribution.
    
    Returns:
    --------
    float: The Cauchy-Schwarz overlap between the two Gaussian distributions.

    Note: this comes from the fact that \int p(x) q(x) dx <= sqrt(\int p(x)^2 dx) * sqrt(\int q(x)^2 dx) -> overlap <= 1 (CS inequality)
    """
    mu = mu.flatten()
    if nu.ndim > 1:
        nu = nu.flatten()
    d = mu.shape[0]
    A  = A + 1e-9*np.eye(d) # add a small jitter for numerical stability
    B = B + 1e-9*np.eye(d)
    Lambda = A + B # sum of the two covariance matrices
    det = np.linalg.det
    dm  = mu - nu # this is the difference mu - nu of the means of the two distributions 
    Mah = dm @ np.linalg.solve(Lambda, dm) # Mahalanobis distance , which is (mu - nu)^T (Lambda)^(-1))(mu - nu) where Lambda is the sum of the two covariance matrices
    return (2.0**(d/2.0)) * (det(A)**0.25) * (det(B)**0.25) / (det(Lambda)**0.5) * np.exp(-0.5*Mah)


def overlap_batch(mu, S, mu_cands, S_cands, threshold=0.15):
    """
    Compute the overlap between a batch of multivariate Gaussian distributions and a candidate Gaussian distribution.
    
    Parameteres:
    ------------
    mu: np.ndarray
        Mean vectors of a candidate of same [1, N_tasks].
    S: np.ndarray
        Covariance matrices of the Candidate, shape (N_tasks, N_tasks).
    mu_cand: np.ndarray
        Mean vector of the previous candidates Gaussian distribution, shape (N_cand, N_tasks).
    S_cand: np.ndarray
        Covariance matrices of the candidates' Gaussian distribution, shape (N_cand, N_tasks, N_tasks).

    Returns:
    --------
    total_overlap: float
        Normalized overlap between the batch of Gaussian distributions and the candidate Gaussian distribution. Lives in [0, 1].
    """
    
    if S.ndim > 2:
        S = S[0]
    results = [overlap_cs_joint(mu, S, mu_cands[i], S_cands[i]) for i in range(mu_cands.shape[0])]
    # get the subset with reasonable overlap
    results_cut = [r for r in results if r > threshold]
    if len(results_cut) == 0:
        total_overlap = 0.0
    else:
        total_overlap = sum(results_cut)/len(results_cut)

    return total_overlap


def augment(cfg: ALConfig, *, seq_id: int, pessimism: bool, log=None) -> None:
    """Augment features with previous selected children up to seq_id-1.
    Parameters:
    ----------
    cfg: ALConfig
        Active learning configuration.
    seq_id: int
        Current sequence ID (1-based).
    pessimism: bool
        Whether to apply pessimism adjustment when using kriging believer.

    Returns:
    -------
    None
    """
    p = cfg.paths
    featurizer = sf.SequenceFeaturizer(cfg.model.lower(), cfg.db_path)
    normalization_stats = load_normalization_stats(p.norm_stats)
    objectives = [cfg.obj1, cfg.obj2]

    feats_total  = pd.read_csv(p.features_norm_csv)

    if cfg.train_model_type == "gpr_multitask":
        labels_total = pd.read_csv(p.labels_norm_csv)
    elif cfg.train_model_type == "gpr_singletask":
        labels_list = []
        for label in objectives:
            y_path = p.labels_csv.with_stem(p.labels_csv.stem + f"_{label}_NORM_{p.tag}")
            y = pd.read_csv(y_path)
            labels_list.append(y)
        labels_total = pd.concat(labels_list, axis=1)


    temp_in  = seq_id > 1
    temp_out = True

    model_bundle = load_models(cfg, temp = temp_in, device='cpu')

    if seq_id <= 1:
        seq_file = cfg.paths.seq_gen_txt
        labels_no_pess = labels_total.copy()
        labels_no_pess.to_csv(p.labels_no_pessimism, index=False)
    else: 
        seq_file = p.seq_gen_temp_txt
        labels_no_pess = pd.read_csv(p.labels_no_pessimism)
    
    column_names = feats_total.columns.tolist()

    child_file = p.ga_children_dir / f"seq_child_{seq_id}.txt"

    seqs = []
    with open(child_file, 'r') as f:
        seq = f.readline().strip()
        seqs.append(seq)

    # Featurize and normalize
    features = []
    # we process one at a time, so the loop is a bit useless, but we can modify later
    for seq in seqs:
        #print(seq, flush=True)
        raw_feats = featurizer.featurize(seq)
        norm_feats = convert_and_normalize_features(np.asarray(raw_feats).reshape(1, -1), train=False, stats=normalization_stats)
        features.append(norm_feats.reshape(-1))

    features = np.array(features)

    new_frame = pd.DataFrame(features, columns=column_names)

    # concatenate with the total features
    feats_new = pd.concat([feats_total, new_frame], ignore_index=True)
    # Save the updated features to the generation folder as temp 
    feats_new.to_csv(p.features_norm_csv, index=False)
    if log:
        log.info(f"Features augmented and saved to {p.features_norm_csv}")
    
    if 'kriging_believer' in cfg.exploration_strategy:
        # Use the GPR model to predict the objectives
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            #preds = likelihood(model(torch.tensor(features).float())).mean.detach().numpy()

            preds, S, sig = predict_for_augmentation(model_bundle, features, cfg, return_std=True)
            
            if pessimism:
                # add pessimism to the predictions
                prev_labels = labels_total.iloc[-seq_id:].values
                prev_feats = feats_total.iloc[-seq_id:].values

                if prev_labels.ndim == 1:
                    prev_labels = prev_labels.copy().reshape(1, -1)

                mu = preds.copy()

                if seq_id > 1:
                    mu_cands, S_cands = predict_for_augmentation(model_bundle, prev_feats, cfg, return_std=False)

                    sign = -1.0 if cfg.front == 'upper' else +1.0

                    penalty = sign*overlap_batch(mu.copy(), S, mu_cands, S_cands) 

                    preds  = mu + penalty * sig if seq_id > 1  else preds 
                else:
                    preds = mu
            else:
                mu = preds.copy()

    elif cfg.exploration_strategy in ['constant_liar_min', 'constant_liar_mean', 'constant_liar_max']:
        # first, get the generation we are at

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            #preds = likelihood(model(torch.tensor(features).float())).mean.detach().numpy()

            mu, S, sig = predict_for_augmentation(model_bundle, features, cfg, return_std=True)
        num_rows = cfg.seed_size + cfg.iteration * cfg.ngen * 2  # 120 is the number of rows in the first generation, 48 is the number of rows added in each subsequent generation
        
        # choose either min, mean or max of the labels before our sequential batch generation as a constant lie

        if cfg.exploration_strategy == 'constant_liar_min':
            preds = labels_total.iloc[:num_rows].min().values
        elif cfg.exploration_strategy == 'constant_liar_mean':
            preds = labels_total.iloc[:num_rows].mean().values
        elif cfg.exploration_strategy == 'constant_liar_max':
            preds = labels_total.iloc[:num_rows].max().values
        preds = np.tile(preds, (features.shape[0], 1))

    new_labels         = pd.DataFrame(preds, columns=labels_total.columns)
    new_labels_no_pess = pd.DataFrame(mu, columns=labels_total.columns)
    
    # Concatenate with the total labels
    labels_total         = pd.concat([labels_total, new_labels], ignore_index=True)
    labels_no_pess_total = pd.concat([labels_no_pess, new_labels_no_pess], ignore_index=True)
    
    if log:
        log.info(f"Labels augmented and saved to {p.labels_norm_csv}")

    # Save the labels without pessimism
    labels_no_pess_total.to_csv(p.labels_no_pessimism, index=False)
    
    # Save the updated labels to the generation folder as temp
    if cfg.train_model_type == "gpr_singletask":
        for idx, label in enumerate(objectives):
            y_path = p.labels_norm_csv.with_stem(p.labels_csv.stem + f"_{label}_NORM_{p.tag}")
            labels_total[[label]].to_csv(y_path, index=False)
    else:
        labels_total.to_csv(p.labels_norm_csv, index=False)

    # now that we have the new files, we can retrain our GPR model on the augmented dataset

    train_X      = torch.tensor(feats_new.values).float()
    labels_total = labels_total[objectives]
    train_y      = torch.tensor(labels_total.values).float()
    
    model_new, likelihood_new = retrain_model(cfg, model_bundle, train_X, train_y)

    if cfg.train_model_type == "gpr_multitask":
        gpr_file = p.gpr_multitask_chkpt(temp = temp_out)
        save_chkpt(gpr_file, model_new)
    elif cfg.train_model_type == "gpr_singletask":
        for idx, label in enumerate(objectives):
            gpr_file = p.gpr_singletask_chkpt([label], temp = temp_out)[0]
            save_chkpt(gpr_file, model_new[idx])
    else:
        raise ValueError(f"Unknown model type: {cfg.train_model_type} not implemented yet.")

    if log:
        log.info(f"Models retrained and saved to {gpr_file}")

    # Save the sequence to the generation folder as temp
    with open(seq_file, 'r') as f:
        existing_seqs = f.readlines()
    existing_seqs = [line.strip() for line in existing_seqs]
    
    existing_seqs.append(seqs[0])  # Append the new sequence

    with open(p.seq_gen_temp_txt, 'w') as f:
        for seq in existing_seqs:
            f.write(seq + '\n')
    
    if log:
        log.info(f"Sequence augmented and saved to {p.seq_gen_temp_txt}")
