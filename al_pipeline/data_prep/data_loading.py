import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

# NOTE: This is *hardcoded*, and should be modified if we change features! 
def convert_and_normalize_features(
    X: np.ndarray,
    train: bool = True,
    stats: dict | None = None,
    ) -> tuple[np.ndarray, dict] | np.ndarray:
    """
    Normalize features for our GPR models

    Parameters:
    -----------
    X : np.ndarray
    train : bool
        If True, compute stats from X and return (X_norm, stats).
        If False, apply given stats and return X_norm.
    stats : dict or None
        Normalization stats from a previous training call:
        {
          "means": {feature_name: float, ...},
          "stds": {feature_name: float, ...},
          "min_L": float,
          "max_L": float,
          "max_S": float,
        }

    Returns:
    --------
     X_norm, stats if train is True else X_norm
     X_norm : np.ndarray
        Normalized features.
    stats: dict
        Normalization statistics to be applied to test set.
    """
    X = np.asarray(X, dtype=np.float32).copy()
    if X.shape[1] != 29:
        raise ValueError(f"Expected X to have 29 features, got {X.shape[1]}")

    # Column indices
    AA_SLICE = slice(0, 20)
    IDX_LEN = 20
    IDX_SCD = 21
    IDX_SHD = 22
    IDX_ABS_Q = 23
    IDX_SUM_LAMBDA = 24
    IDX_BEADS_POS = 25
    IDX_BEADS_NEG = 26
    IDX_SHAN_ENT = 27
    IDX_MOL_WT = 28

    # Get lengths of sequences
    length = X[:, IDX_LEN]  # shape (N,)

    if np.any(length <= 0):
        raise ValueError("Found non-positive sequence length; cannot normalize by length.")

    # AA counts -> frequencies
    X[:, AA_SLICE] = X[:, AA_SLICE] / length[:, None]

    # Divide selected features by length as fractions per residue
    X[:, IDX_BEADS_POS]   = X[:, IDX_BEADS_POS]   / length
    X[:, IDX_BEADS_NEG]   = X[:, IDX_BEADS_NEG]   / length
    X[:, IDX_ABS_Q]       = X[:, IDX_ABS_Q]       / length
    X[:, IDX_SUM_LAMBDA]  = X[:, IDX_SUM_LAMBDA]  / length
    X[:, IDX_MOL_WT]      = X[:, IDX_MOL_WT]      / length

    # 2) Standard and min-max normalization
    # Features to standardize
    std_features = {
        "beads_pos":   IDX_BEADS_POS,
        "beads_neg":   IDX_BEADS_NEG,
        "sum_lambda":  IDX_SUM_LAMBDA,
        "mol_wt":      IDX_MOL_WT,
        "SHD":         IDX_SHD,
        "SCD":         IDX_SCD,
        "abs_net_q":   IDX_ABS_Q,
    }

    if train:
        means = {}
        stds = {}

        # standardize selected features
        for name, idx in std_features.items():
            col = X[:, idx]
            mu = float(col.mean())
            sigma = float(col.std(ddof=0))
            if sigma == 0.0:
                sigma = 1.0  # avoid div by zero; everything is identical anyway
            X[:, idx] = (col - mu) / sigma
            means[name] = mu
            stds[name] = sigma

        # min-max for length
        L = X[:, IDX_LEN]
        min_L = float(L.min())
        max_L = float(L.max())
        denom_L = max_L - min_L if max_L > min_L else 1.0
        X[:, IDX_LEN] = (L - min_L) / denom_L

        # Shannon entropy S_scaled = S/max(S) - 1
        S = X[:, IDX_SHAN_ENT]
        max_S = float(S.max()) if S.size > 0 else 1.0
        if max_S == 0.0:
            max_S = 1.0
        X[:, IDX_SHAN_ENT] = S / max_S - 1.0

        stats_out = {
            "means": means,
            "stds": stds,
            "min_L": min_L,
            "max_L": max_L,
            "max_S": max_S,
        }
        return X.astype(np.float32), stats_out

    else:
        if stats is None:
            raise ValueError("stats must be provided when train=False")

        means = stats["means"]
        stds = stats["stds"]
        min_L = stats["min_L"]
        max_L = stats["max_L"]
        max_S = stats["max_S"]

        # standardize using provided stats
        for name, idx in std_features.items():
            mu = means[name]
            sigma = stds[name] if stds[name] != 0.0 else 1.0
            X[:, idx] = (X[:, idx] - mu) / sigma

        # min-max for length with training stats
        L = X[:, IDX_LEN]
        denom_L = max_L - min_L if max_L > min_L else 1.0
        X[:, IDX_LEN] = (L - min_L) / denom_L

        # Shannon with training max_S
        S = X[:, IDX_SHAN_ENT]
        if max_S == 0.0:
            max_S = 1.0
        X[:, IDX_SHAN_ENT] = S / max_S - 1.0

        return X.astype(np.float32)


# Function to load and prepare the dataset
def load_dataset(features_file, labels_file, label_columns = ['exp_density', 'diff'], model=None):
    """
    Load the dataset for training.
    
    Parameters:
    features_file (str): Path to the CSV file containing the feature data.
    labels_file (str): Path to the CSV file containing the labels.
    label_columns (list of str): The names of the label columns to extract from the labels file.
    
    Returns:
    Dataset: A PyTorch dataset with the features and labels ready for training.
    """
    # Load and normalize features
    features_df = pd.read_csv(features_file)
    
    # Load labels
    labels_df = pd.read_csv(labels_file)

    # Make sure we have no nan values for training...
    labels_nonan = labels_df.dropna(subset=label_columns)
    feats_nonan = features_df[features_df.index.isin(labels_nonan.index)]
    
    # Extract labels for the specified property
    labels = labels_nonan[label_columns].values
    features = feats_nonan.values
    

    #Create dataset if we are using dense neural net
    if model == 'dnn':
        dataset = ProteinDataset(features, labels)
        return dataset
    else:
        
        return features, labels

class ProteinDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

