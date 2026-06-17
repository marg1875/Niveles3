"""
Feature subset and channel subset handling for classification.

Functions to:
1. Load features and filter by channel subsets (averaging per-channel features)
2. Filter by feature subsets (Delta, Delta-noVar, etc.)
3. Handle stability-based filtering (Delta-Stable)
"""
import numpy as np
import pandas as pd

import config as cfg

_FRACTAL_METHODS = ["RS", "Higuchi", "DFA", "Variogram", "Average", "HO", "HRS_p64", "HV"]

def get_channel_subset_features(df: pd.DataFrame, channel_config: str) -> pd.DataFrame:
    """Return a DataFrame with features averaged over a specific channel subset.

    For channel_configs with >1 channel: computes spatial average of per-channel
    features (Basal_{ch}_{method}, Active_{ch}_{method}, Delta_{ch}_{method})
    and returns them as Basal_{method}, Active_{method}, Delta_{method}.

    For "All-16": returns the pre-averaged columns directly (faster).
    For single-channel configs: extracts the per-channel values directly.

    Args:
        df: Full features DataFrame with per-channel columns.
        channel_config: Key from config.CHANNEL_SUBSETS.

    Returns:
        DataFrame with spatial-averaged features for the subset.
    """
    if channel_config not in cfg.CHANNEL_SUBSETS:
        raise ValueError(f"Unknown channel config: {channel_config}. "
                         f"Options: {list(cfg.CHANNEL_SUBSETS.keys())}")

    indices = cfg.CHANNEL_SUBSETS[channel_config]
    channels = [cfg.CHANNEL_NAMES[i] for i in indices]

    # If All-16, use pre-averaged columns directly
    if channel_config == "All-16" or len(indices) == 16:
        return df.copy()

    result = df.copy()

    # For single channel: rename per-channel columns to averaged names
    if len(indices) == 1:
        ch = channels[0]
        for ftype in ["Basal", "Active", "Delta"]:
            for method in _FRACTAL_METHODS:
                src_col = f"{ftype}_{ch}_{method}"
                dst_col = f"{ftype}_{method}"
                if src_col in result.columns:
                    result[dst_col] = result[src_col]
        return result

    # For multiple channels: average per-channel columns
    for ftype in ["Basal", "Active", "Delta"]:
        for method in _FRACTAL_METHODS:
            cols_to_avg = [f"{ftype}_{ch}_{method}" for ch in channels
                           if f"{ftype}_{ch}_{method}" in result.columns]
            if cols_to_avg:
                result[f"{ftype}_{method}"] = result[cols_to_avg].mean(axis=1)

    return result


def get_feature_subset(df: pd.DataFrame, feature_subset: str) -> tuple:
    """Extract feature columns and labels for a given feature subset.

    Args:
        df: DataFrame with averaged features (from get_channel_subset_features).
        feature_subset: Key from config.CLASS_FEATURE_SUBSETS.

    Returns:
        (X, y, feature_columns, filtered_df) where filtered_df has aligned Patient col.
    """
    y = df["MVR_Class"].values

    if feature_subset == "Delta-Stable":
        # Filter to only Delta features with CV < 0.10
        stable_cols = []
        for method in _FRACTAL_METHODS:
            cv_col = f"{method}_cv"
            if cv_col in df.columns:
                stable_idx = df[cv_col] < cfg.CV_STABLE_THRESHOLD
                stable_cols.append(stable_idx)
        if stable_cols:
            # Epoc must be stable across all methods
            stable_mask = np.all(np.column_stack(stable_cols), axis=1)
            df = df[stable_mask].copy()
            y = df["MVR_Class"].values
        feature_cols = ["Delta_RS", "Delta_Higuchi", "Delta_DFA", "Delta_Variogram",
                        "Delta_Average"]
    elif feature_subset == "All":
        # All numeric columns except metadata/stability/categorical
        exclude_patterns = ["Patient", "Session", "Paradigm", "Event_Time",
                           "_window_sec", "_stability", "MVR_Class"]
        feature_cols = [c for c in df.columns
                       if df[c].dtype in ("float64", "float32", "int64")
                       and not any(p in c for p in exclude_patterns)]
    else:
        feature_cols = cfg.CLASS_FEATURE_SUBSETS.get(feature_subset)
        if feature_cols is None:
            raise ValueError(f"Unknown feature subset: {feature_subset}")

    # Keep only existing feature columns
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].values

    # Replace inf with NaN, then NaN with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, feature_cols, df


def filter_2class(X, y, df, mvr_levels=(10, 40)):
    """Filter data to only two classes (e.g., 10% and 40% MVR)."""
    mask = np.isin(y, mvr_levels)
    return X[mask], y[mask], df[mask]
