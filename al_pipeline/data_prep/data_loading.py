import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np


# ----------------------------------------------------------------------
# DataFrame-based feature pipeline (used by MoE per-expert training)
# ----------------------------------------------------------------------
#
# The existing `convert_and_normalize_features` operates on a 29-column numpy
# array with hardcoded indices and does convert + fit + apply in a single call.
# That's fine for the global-GPR path that trains on every row at once, but the
# MoE workflow needs to fit a feature normalizer SEPARATELY per expert (PS
# rows, nonPS rows, all rows). For that the three steps must be callable
# independently:
#
#   df_conv     = convert_features(features_df)
#   stats_ps    = fit_feature_normalizer(df_conv.loc[is_ps])
#   df_norm_ps  = apply_feature_normalizer(df_conv.loc[is_ps], stats_ps)
#
# These DataFrame functions are numerically equivalent to the numpy path: a
# round-trip through `df.values` produces the same normalized array. Verified
# by tests/al_pipeline/test_feature_pipeline.py.

_FEATURES_TO_STANDARDIZE = (
    "beads(+)", "beads(-)", "sum lambda", "mol wt", "SHD", "SCD", "|net charge|",
)
_FEATURES_DIVIDED_BY_LENGTH = (
    "beads(+)", "beads(-)", "|net charge|", "sum lambda", "mol wt",
)


def convert_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Length-normalize raw features WITHOUT fitting any standardizer.

    The first 20 columns (AA counts) become fractions, and a handful of
    aggregate features get divided by length too. No normalization stats are
    fit here — that's `fit_feature_normalizer`'s job.

    All columns are cast to float32 to keep the entire pipeline in single
    precision (consistent with the numpy path and with downstream GPyTorch
    training tensors).

    Parameters
    ----------
    features_df : pd.DataFrame
        Raw featurizer output. Must contain column 'length' and the standard
        29-column feature layout produced by SequenceFeaturizer.featurize_many.

    Returns
    -------
    pd.DataFrame
        Same shape and columns as the input, with the length-normalized
        operations applied, dtype float32. Input is not mutated.
    """
    df = features_df.astype(np.float32, copy=True)
    # First 20 columns are AA counts (in AMINO_ACIDS order).
    for col in df.columns[:20]:
        df[col] = df[col] / df["length"]
    for col in _FEATURES_DIVIDED_BY_LENGTH:
        df[col] = df[col] / df["length"]
    return df


def fit_feature_normalizer(features_converted_df: pd.DataFrame) -> dict:
    """
    Fit per-feature normalization statistics on already-converted features.

    Standardization uses population std (ddof=0) — matches the existing numpy
    `convert_and_normalize_features` and treats the AL generations as the
    population, not a sample of a larger one. Stats are stored as float32 so
    a subsequent `apply_feature_normalizer` stays in single precision.

    Parameters
    ----------
    features_converted_df : pd.DataFrame
        Output of `convert_features`. Standardization stats are fit on whatever
        subset of rows is passed (e.g. PS rows only for the PS expert).

    Returns
    -------
    dict
        Per-feature stats. Each entry has a 'type' key, plus the stats needed
        to apply that type:
          - {'type': 'standard', 'mean': float32, 'std': float32}
          - {'type': 'minmax',   'min':  float32, 'max': float32, 'range': float32}
          - {'type': 'shanent',  'max':  float32}
    """
    stats: dict = {}

    for feat in _FEATURES_TO_STANDARDIZE:
        col = features_converted_df[feat]
        mean = float(col.mean())
        # population std (ddof=0) — matches convert_and_normalize_features
        std = float(col.std(ddof=0))
        if pd.isna(mean):
            mean = 0.0
        if pd.isna(std) or std == 0:
            std = 1.0
        stats[feat] = {
            "type": "standard",
            "mean": np.float32(mean),
            "std":  np.float32(std),
        }

    min_L = float(features_converted_df["length"].min())
    max_L = float(features_converted_df["length"].max())
    range_L = max_L - min_L
    if range_L == 0 or pd.isna(range_L):
        range_L = 1.0
    stats["length"] = {
        "type": "minmax",
        "min":   np.float32(min_L),
        "max":   np.float32(max_L),
        "range": np.float32(range_L),
    }

    max_S = float(features_converted_df["shan ent"].max())
    if pd.isna(max_S) or max_S == 0:
        max_S = 1.0
    stats["shan ent"] = {"type": "shanent", "max": np.float32(max_S)}

    return stats


def apply_feature_normalizer(
    features_converted_df: pd.DataFrame,
    stats: dict,
) -> pd.DataFrame:
    """
    Apply previously-fit normalization stats to converted features.

    All arithmetic stays in float32 (stats are float32, the input is float32
    from `convert_features`). The output is cast to float32 explicitly to
    guard against accidental upcasting if a caller passes float64 input.

    Parameters
    ----------
    features_converted_df : pd.DataFrame
        Output of `convert_features` (length-normalized, not standardized).
    stats : dict
        Output of `fit_feature_normalizer`.

    Returns
    -------
    pd.DataFrame
        Same shape as input, fully normalized, dtype float32. Input is not
        mutated.
    """
    df = features_converted_df.astype(np.float32, copy=True)
    for feat, s in stats.items():
        kind = s["type"]
        if kind == "standard":
            df[feat] = (df[feat] - s["mean"]) / s["std"]
        elif kind == "minmax":
            df[feat] = (df[feat] - s["min"]) / s["range"]
        elif kind == "shanent":
            df[feat] = df[feat] / s["max"] - np.float32(1.0)
        else:
            raise ValueError(f"Unknown normalizer type: {kind!r} for feature {feat!r}")
    return df


# ----------------------------------------------------------------------
# Numpy-array pipeline (existing global-GPR path; unchanged behavior)
# ----------------------------------------------------------------------

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

