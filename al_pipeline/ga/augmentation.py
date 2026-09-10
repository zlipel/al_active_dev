from __future__ import annotations

from al_pipeline.core.config import ALConfig
from al_pipeline.training.ml_models import MultitaskGPRegressionModel, GPRegressionModel
from al_pipeline.training.kfold_training import save_chkpt
from al_pipeline.data_prep.data_loading import (
    apply_feature_normalizer, convert_features, convert_and_normalize_features, load_dataset,
)
from al_pipeline.surrogates import (
    build_rf_features, classifier_p_ps, make_surrogate, save_rf_bundle,
)
from .ga_utils import load_moe_bundle, load_normalization_stats, load_models
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

def predict_for_augmentation(surrogate, features_raw_df: pd.DataFrame, return_std: bool = False):
    """
    Uniform prediction path for kriging-believer + constant-liar augmentation.

    Takes any `Surrogate` (global multitask / singletask GPR or MoE) and raw
    features as a DataFrame. Returns per-candidate means, per-candidate (2, 2)
    covariance, and optionally per-candidate marginal stds — all in the
    normalized objective z-space that the AL loop's Pareto / EHVI machinery
    already operates in.

    The pessimism formula (see `overlap_batch`) is agnostic to surrogate type;
    all it needs is `mu` and the joint `cov`. Under MoE the cov is the
    mixture cov from `MoEPoolPosterior.covariance`; under global multitask
    it's the per-candidate block from the GP's joint posterior; under
    single-task it degrades to a diagonal fallback.
    """
    pool = surrogate.predict_pool(features_raw_df)
    mu = pool.means
    cov = pool.covariance   # (N, 2, 2)
    if return_std:
        return mu, cov, pool.stds
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


def _load_surrogate_for_augment(cfg: ALConfig, temp_in: bool):
    """Load bundle + build Surrogate. Returns (surrogate, model_bundle_or_moe_bundle)."""
    if cfg.train_model_type == "moe":
        moe_bundle = load_moe_bundle(cfg, temp=temp_in)
        surrogate = make_surrogate(
            cfg, moe_bundle=moe_bundle,
            moe_policy=cfg.moe_policy, moe_threshold=cfg.moe_threshold,
        )
        return surrogate, moe_bundle
    else:
        normalization_stats = load_normalization_stats(cfg.paths.norm_stats)
        model_bundle = load_models(cfg, temp=temp_in, device="cpu")
        surrogate = make_surrogate(
            cfg, model_bundle=model_bundle, normalization_stats=normalization_stats,
        )
        return surrogate, model_bundle


def _reindex_expert(assigned, new_child_raw_df: pd.DataFrame,
                     new_child_z: np.ndarray, lr: float):
    """
    Re-condition ONE MoE expert on an augmented train set (kriging-believer
    hard-gate assignment). Freezes hyperparameters modulo one Adam step —
    same "1 epoch" recipe the global multitask path uses.

    Mutates `assigned.model` and `assigned.likelihood` in place. Returns the
    expanded (train_x, train_y) tensors so the caller can persist them into
    the temp checkpoint.
    """
    # Current expert state (before augmentation)
    current_train_x = assigned.model.train_inputs[0]
    current_train_y = assigned.model.train_targets

    # Normalize the new child through THIS expert's own feature normalizer
    feat_conv = convert_features(new_child_raw_df[assigned.feature_columns])
    feat_norm = apply_feature_normalizer(feat_conv, assigned.feature_normalizer_stats)
    new_x = torch.tensor(feat_norm.to_numpy(), dtype=torch.float32)
    new_y = torch.tensor(np.asarray(new_child_z).reshape(-1, 2), dtype=torch.float32)

    train_x_new = torch.cat([current_train_x, new_x], dim=0)
    train_y_new = torch.cat([current_train_y, new_y], dim=0)

    # Warm-start ExactGP re-instantiation with the expanded training set —
    # same shape as _retrain_model_gpr_multitask, just on the assigned expert.
    likelihood_new = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    model_new = MultitaskGPRegressionModel(train_x_new, train_y_new, likelihood_new, num_tasks=2)
    likelihood_new.load_state_dict(assigned.likelihood.state_dict())
    model_new.load_state_dict(assigned.model.state_dict())

    model_new.train(); likelihood_new.train()
    optimizer = torch.optim.Adam(model_new.parameters(), lr=lr / 100.0)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_new, model_new)
    optimizer.zero_grad()
    output = model_new(train_x_new)
    loss = -mll(output, train_y_new)
    loss.backward()
    optimizer.step()
    model_new.eval(); likelihood_new.eval()

    assigned.model = model_new
    assigned.likelihood = likelihood_new
    return train_x_new, train_y_new


def _save_moe_temp(cfg: ALConfig, moe_bundle):
    """
    Persist MoE temp checkpoints for the next seq_id.

    Both experts are reconditioned in place (kriging-believer conditions the
    whole mixture, not just the gated expert), so each stores its own expanded
    train tensors directly — temp checkpoints don't need to be re-derivable from
    CSVs, the batch throws them away once real labels come in. The RF bundle is
    saved verbatim (frozen during batch generation).
    """
    p = cfg.paths
    common = {
        "model_name":         cfg.model,
        "iteration":          cfg.iteration,
        "label_scaler_scope": "all",
    }

    moe_bundle.ps_expert.save_checkpoint(
        str(p.moe_ps_chkpt(temp=True)),
        regime="ps",
        original_indices=[],   # ignored when direct tensors are present
        train_x_direct=moe_bundle.ps_expert.model.train_inputs[0],
        train_y_direct=moe_bundle.ps_expert.model.train_targets,
        **common,
    )
    moe_bundle.nonps_expert.save_checkpoint(
        str(p.moe_nonps_chkpt(temp=True)),
        regime="nonps",
        original_indices=[],
        train_x_direct=moe_bundle.nonps_expert.model.train_inputs[0],
        train_y_direct=moe_bundle.nonps_expert.model.train_targets,
        **common,
    )

    # RF: verbatim copy of whatever bundle we loaded (base or temp).
    rf_bundle = moe_bundle.rf_bundle
    save_rf_bundle(
        str(p.moe_rf_bundle(temp=True)),
        moe_bundle.rf,
        rf_raw_feature_columns=rf_bundle["rf_raw_feature_columns"],
        rf_converted_feature_columns=rf_bundle["rf_converted_feature_columns"],
        ps_definition=rf_bundle.get("ps_definition", ""),
        random_state=rf_bundle.get("random_state", 0),
        threshold=rf_bundle.get("threshold", 0.5),
        model_name=cfg.model,
        iteration=cfg.iteration,
        transform=rf_bundle.get("transform", cfg.transform),
        label_scaler_scope="all",
        best_params=rf_bundle.get("best_params"),
    )


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

    feats_total = pd.read_csv(p.features_norm_csv)

    if cfg.train_model_type in ("gpr_multitask", "moe"):
        labels_total = pd.read_csv(p.labels_norm_csv)
    elif cfg.train_model_type == "gpr_singletask":
        labels_list = []
        for label in objectives:
            y_path = p.labels_csv.with_stem(p.labels_csv.stem + f"_{label}_NORM_{p.tag}")
            y = pd.read_csv(y_path)
            labels_list.append(y)
        labels_total = pd.concat(labels_list, axis=1)
    else:
        raise ValueError(f"Unknown train_model_type: {cfg.train_model_type}")

    temp_in = seq_id > 1
    temp_out = True

    surrogate, bundle = _load_surrogate_for_augment(cfg, temp_in=temp_in)

    if seq_id <= 1:
        seq_file = p.seq_gen_txt
        labels_no_pess = labels_total.copy()
        labels_no_pess.to_csv(p.labels_no_pessimism, index=False)
    else:
        seq_file = p.seq_gen_temp_txt
        labels_no_pess = pd.read_csv(p.labels_no_pessimism)

    column_names = feats_total.columns.tolist()
    child_file = p.ga_children_dir / f"seq_child_{seq_id}.txt"

    with open(child_file, "r") as f:
        seqs = [f.readline().strip()]

    # Raw + normalized features for the new child. Raw feeds the surrogate
    # (uniform contract); normalized feeds the on-disk features_norm_csv
    # append (used by any downstream tool that reads global-normalized data).
    raw_feats_arr = np.asarray(featurizer.featurize(seqs[0])).reshape(1, -1)
    raw_feats_df = pd.DataFrame(raw_feats_arr, columns=column_names)
    norm_feats_arr = convert_and_normalize_features(raw_feats_arr, train=False, stats=normalization_stats)
    new_frame = pd.DataFrame(norm_feats_arr, columns=column_names)

    feats_new = pd.concat([feats_total, new_frame], ignore_index=True)
    feats_new.to_csv(p.features_norm_csv, index=False)
    if log:
        log.info(f"Features augmented and saved to {p.features_norm_csv}")

    # Pessimism/no-pessimism split. Same for all surrogate types now that
    # predict_for_augmentation is uniform.
    if "kriging_believer" in cfg.exploration_strategy:
        preds, S, sig = predict_for_augmentation(surrogate, raw_feats_df, return_std=True)
        if pessimism and seq_id > 1:
            # Earlier children in this batch, RAW-featurized. features_norm_csv
            # holds GLOBALLY-NORMALIZED rows, but the surrogate expects RAW
            # features (it normalizes internally — per-regime for the MoE experts,
            # and the RF gate re-derives its own convert_features matrix). Feeding
            # normalized rows as "raw" pushes convert_features' log/ratio terms to
            # inf and sklearn rejects the RF input. Re-featurize from the picked
            # sequences instead, matching the current child's raw path.
            prev_seqs = []
            for i in range(1, seq_id):
                with open(p.ga_children_dir / f"seq_child_{i}.txt", "r") as cf:
                    prev_seqs.append(cf.readline().strip())
            prev_raw_arr = np.asarray([featurizer.featurize(s) for s in prev_seqs])
            prev_feats_raw_df = pd.DataFrame(prev_raw_arr, columns=column_names)
            mu_cands, S_cands = predict_for_augmentation(surrogate, prev_feats_raw_df, return_std=False)
            sign = -1.0 if cfg.front == "upper" else +1.0
            penalty = sign * overlap_batch(preds.copy(), S, mu_cands, S_cands)
            mu = preds.copy()
            preds = preds + penalty * sig
        else:
            mu = preds.copy()

    elif cfg.exploration_strategy in ("constant_liar_min", "constant_liar_mean", "constant_liar_max"):
        mu, S, sig = predict_for_augmentation(surrogate, raw_feats_df, return_std=True)
        num_rows = cfg.seed_size + cfg.iteration * cfg.ngen * 2
        base = labels_total.iloc[:num_rows]
        if cfg.exploration_strategy == "constant_liar_min":
            const = base.min().values
        elif cfg.exploration_strategy == "constant_liar_mean":
            const = base.mean().values
        else:
            const = base.max().values
        preds = np.tile(const, (raw_feats_arr.shape[0], 1))
    else:
        raise ValueError(
            f"augment() only supports kriging_believer / constant_liar_*, got {cfg.exploration_strategy!r}"
        )

    new_labels         = pd.DataFrame(preds, columns=labels_total.columns)
    new_labels_no_pess = pd.DataFrame(mu, columns=labels_total.columns)
    labels_total         = pd.concat([labels_total, new_labels], ignore_index=True)
    labels_no_pess_total = pd.concat([labels_no_pess, new_labels_no_pess], ignore_index=True)

    if log:
        log.info(f"Labels augmented and saved to {p.labels_norm_csv}")
    labels_no_pess_total.to_csv(p.labels_no_pessimism, index=False)

    if cfg.train_model_type == "gpr_singletask":
        for label in objectives:
            y_path = p.labels_norm_csv.with_stem(p.labels_csv.stem + f"_{label}_NORM_{p.tag}")
            labels_total[[label]].to_csv(y_path, index=False)
    else:
        labels_total.to_csv(p.labels_norm_csv, index=False)

    # Retrain / re-condition
    if cfg.train_model_type == "moe":
        # Kriging-believer for the soft mixture: recondition BOTH experts on the
        # SAME believed label so they agree at the fantasy point x*. The EHVI is
        # taken over the moment-matched mixture whose variance carries a
        # between-expert disagreement term p(1-p)(mu_PS - mu_nonPS)^2. Conditioning
        # only the gated expert leaves that term (and the other expert's variance)
        # intact, so the mixture EHVI barely moves at x* and the batch re-picks the
        # same optimum. Driving mu_PS(x*) and mu_nonPS(x*) to the same believed
        # value collapses the disagreement term (and both within-variances) at x*,
        # so the mixture EHVI actually drops and the batch diversifies.
        #
        # Believed label = preds[-1]: pessimism-adjusted when --pessimism (seq_id>1),
        # else the plain posterior mean — the same value written to labels_norm_csv.
        # The RF gate stays frozen; its routing is logged for provenance only.
        X_rf, _ = build_rf_features(
            raw_feats_df, bundle.rf_raw_feature_columns, bundle.rf_converted_feature_columns,
        )
        p_ps_child = float(classifier_p_ps(bundle.rf, X_rf)[0])
        regime = "ps" if p_ps_child >= cfg.moe_threshold else "nonps"

        _reindex_expert(bundle.ps_expert, raw_feats_df, preds[-1], lr=cfg.learning_rate)
        _reindex_expert(bundle.nonps_expert, raw_feats_df, preds[-1], lr=cfg.learning_rate)
        _save_moe_temp(cfg, bundle)
        if log:
            log.info(
                f"MoE seq_id={seq_id}: reconditioned BOTH experts on the believed "
                f"label (gate routed x* to {regime!r}, p_ps={p_ps_child:.3f}); "
                f"temp checkpoints saved."
            )
    else:
        # Global GPR: preserve the pre-refactor pattern exactly.
        train_X = torch.tensor(feats_new.values).float()
        train_y = torch.tensor(labels_total[objectives].values).float()
        model_new, _lik_new = retrain_model(cfg, bundle, train_X, train_y)

        if cfg.train_model_type == "gpr_multitask":
            gpr_file = p.gpr_multitask_chkpt(temp=temp_out)
            save_chkpt(gpr_file, model_new)
        elif cfg.train_model_type == "gpr_singletask":
            for idx, label in enumerate(objectives):
                gpr_file = p.gpr_singletask_chkpt([label], temp=temp_out)[0]
                save_chkpt(gpr_file, model_new[idx])
        if log:
            log.info("Models retrained and saved.")

    # Sequence bookkeeping — append the child to `seq_gen_temp_txt` so the
    # next seq_id's get_parents call sees the augmented set. Universal for
    # both global and MoE paths.
    with open(seq_file, "r") as f:
        existing_seqs = [ln.strip() for ln in f.readlines()]
    existing_seqs.append(seqs[0])
    with open(p.seq_gen_temp_txt, "w") as f:
        for seq in existing_seqs:
            f.write(seq + "\n")
    
    if log:
        log.info(f"Sequence augmented and saved to {p.seq_gen_temp_txt}")
