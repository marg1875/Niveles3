"""
CSV and ARFF export for extracted features.

Exports:
- Per-file CSV with all feature columns
- Global consolidated CSV (all files combined)
- ARFF format for WEKA (with configurable feature subset)
"""
import os
import numpy as np
import pandas as pd

import config as cfg

# Methods and types for per-channel features
_FRACTAL_METHODS = ["RS", "Higuchi", "DFA", "Variogram", "HO", "HRS_p64", "HV"]
_MARTINEZ_METHODS = ["HO", "HV"] + [f"HRS_p{p}" for p in cfg.HRS_PARTITION_VALUES]
_FEATURE_TYPES = ["Basal", "Active", "Delta"]

_SPECTRAL_COLS = ["delta_power", "theta_power", "alpha_power", "beta_power", "gamma_power",
                  "delta_erd", "theta_erd", "alpha_erd", "beta_erd", "gamma_erd",
                  "alpha_beta_ratio", "mu_beta_ratio", "spectral_entropy"]

# Build per-channel column names: e.g. Basal_Fp1_RS
_PER_CHANNEL_COLS = []
for ftype in _FEATURE_TYPES:
    for ch in cfg.CHANNEL_NAMES:
        for method in _FRACTAL_METHODS:
            _PER_CHANNEL_COLS.append(f"{ftype}_{ch}_{method}")

# Feature column order for output CSV
FEATURE_COLUMNS = [
    # Metadata
    "Patient", "Session", "Month", "Paradigm", "MVR_Class", "Event_Time_sec",
    # Basal features (spatial average)
    "Basal_RS", "Basal_Higuchi", "Basal_DFA", "Basal_Variogram", "Basal_Average",
    "Basal_HO", "Basal_HRS_p64", "Basal_HV",
] + [f"Basal_{m}" for m in _MARTINEZ_METHODS if m not in ("HO", "HRS_p64", "HV")] + [
    # Active features (spatial average, primary = max window)
    "Active_RS", "Active_Higuchi", "Active_DFA", "Active_Variogram", "Active_Average",
    "Active_HO", "Active_HRS_p64", "Active_HV",
] + [f"Active_{m}" for m in _MARTINEZ_METHODS if m not in ("HO", "HRS_p64", "HV")] + [
    # Delta = Active - Basal (spatial average)
    "Delta_RS", "Delta_Higuchi", "Delta_DFA", "Delta_Variogram", "Delta_Average",
    "Delta_HO", "Delta_HRS_p64", "Delta_HV",
] + [f"Delta_{m}" for m in _MARTINEZ_METHODS if m not in ("HO", "HRS_p64", "HV")] + [
    # Stability metrics
    "RS_window_sec", "RS_cv", "RS_stability",
    "Higuchi_window_sec", "Higuchi_cv", "Higuchi_stability",
    "DFA_window_sec", "DFA_cv", "DFA_stability",
    "Variogram_window_sec", "Variogram_cv", "Variogram_stability",
    "Average_window_sec", "Average_cv", "Average_stability",
] + _PER_CHANNEL_COLS + _SPECTRAL_COLS


def build_feature_record(patient: str, month: int, paradigm: str,
                          mvr_class: int, event_time_sec: float,
                          basal_global: np.ndarray,
                          stable_features: dict,
                          original_filename: str,
                          per_channel_features: dict = None,
                          martinez_features: dict = None,
                          spectral_features: dict = None) -> dict:
    """Assemble a single feature row (epoch) from all components.

    Args:
        patient: Patient ID.
        month: Month number (1, 3, 6).
        paradigm: "ASYNCHRONOUS" or "SYNCHRONOUS".
        mvr_class: MVR level (0 for basal, 10 or 40 for active).
        event_time_sec: Time of the event in seconds from recording start.
        basal_global: (5,) array of basal features [RS, Hig, DFA, Var, Avg]
        stable_features: Dict from compute_stable_features().
        original_filename: Original .mat filename.
        per_channel_features: Optional dict with per-channel feature arrays.
            Keys: "basal_per_ch" (n_ch x n_methods), "active_per_ch" (n_ch x n_methods).
        martinez_features: Optional dict with Martinez method values.
            Keys: "basal" and "active", each a dict with keys from _MARTINEZ_METHODS.
        spectral_features: Optional dict from compute_spectral_features().

    Returns:
        Dict with all columns from FEATURE_COLUMNS.
    """
    if basal_global is None:
        basal_global = np.full(5, np.nan)

    record = {
        "Patient": patient,
        "Session": f"Month{month}",
        "Month": month,
        "Paradigm": paradigm,
        "MVR_Class": mvr_class,
        "Event_Time_sec": round(event_time_sec, 3),
        "Basal_RS": basal_global[0],
        "Basal_Higuchi": basal_global[1],
        "Basal_DFA": basal_global[2],
        "Basal_Variogram": basal_global[3],
        "Basal_Average": basal_global[4],
    }

    # Martinez method spatial averages
    if martinez_features is not None:
        basal_m = martinez_features.get("basal", {})
        active_m = martinez_features.get("active", {})
        for method in _MARTINEZ_METHODS:
            b_val = basal_m.get(method, np.nan)
            a_val = active_m.get(method, np.nan)
            record[f"Basal_{method}"] = b_val
            record[f"Active_{method}"] = a_val
            if not (np.isnan(a_val) or np.isnan(b_val)):
                record[f"Delta_{method}"] = a_val - b_val
            else:
                record[f"Delta_{method}"] = np.nan

    # Active features — extracted from stable_features
    active_values = []
    for method in ["RS", "Higuchi", "DFA", "Variogram", "Average"]:
        val = stable_features.get(f"{method}_value", np.nan)
        record[f"Active_{method}"] = val
        active_values.append(val)

        record[f"{method}_window_sec"] = stable_features.get(f"{method}_window_sec", 0.0)
        record[f"{method}_cv"] = stable_features.get(f"{method}_cv", np.nan)
        record[f"{method}_stability"] = stable_features.get(f"{method}_stability", "no_data")

    # Delta = Active - Basal
    for i, method in enumerate(["RS", "Higuchi", "DFA", "Variogram", "Average"]):
        active = active_values[i] if not np.isnan(active_values[i]) else np.nan
        basal = basal_global[i] if not np.isnan(basal_global[i]) else np.nan
        record[f"Delta_{method}"] = active - basal if not (np.isnan(active) or np.isnan(basal)) else np.nan

    # Per-channel features
    if per_channel_features is not None:
        basal_pc = per_channel_features.get("basal_per_ch")  # (n_ch, n_methods)
        active_pc = per_channel_features.get("active_per_ch")  # (n_ch, n_methods)

        n_ch = cfg.N_CHANNELS
        n_methods = len(_FRACTAL_METHODS)
        if basal_pc is None:
            basal_pc = np.full((n_ch, n_methods), np.nan)
        if active_pc is None:
            active_pc = np.full((n_ch, n_methods), np.nan)

        for ch_idx, ch_name in enumerate(cfg.CHANNEL_NAMES):
            for m_idx, method in enumerate(_FRACTAL_METHODS):
                if m_idx < basal_pc.shape[1]:
                    b_val = basal_pc[ch_idx, m_idx]
                else:
                    b_val = np.nan
                if m_idx < active_pc.shape[1]:
                    a_val = active_pc[ch_idx, m_idx]
                else:
                    a_val = np.nan
                record[f"Basal_{ch_name}_{method}"] = b_val
                record[f"Active_{ch_name}_{method}"] = a_val
                # Delta per-channel
                if not (np.isnan(a_val) or np.isnan(b_val)):
                    record[f"Delta_{ch_name}_{method}"] = a_val - b_val
                else:
                    record[f"Delta_{ch_name}_{method}"] = np.nan

    # Spectral features
    if spectral_features is not None:
        for col in _SPECTRAL_COLS:
            record[col] = spectral_features.get(col, np.nan)

    return record


def save_epoch_csv(epochs: list, filepath: str):
    """Save a list of epoch dicts to CSV."""
    if not epochs:
        return
    df = pd.DataFrame(epochs)
    # Reorder columns
    existing_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    df = df[existing_cols]
    df.to_csv(filepath, index=False, encoding="utf-8")


def load_global_features(csv_path: str = None) -> pd.DataFrame:
    """Load the global consolidated features CSV."""
    if csv_path is None:
        csv_path = os.path.join(cfg.FEATURES_DIR, "Global_All_Features.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Global features not found: {csv_path}")
    return pd.read_csv(csv_path)


def export_arff(df: pd.DataFrame, output_path: str,
                feature_columns: list = None,
                class_column: str = "MVR_Class"):
    """Export DataFrame to WEKA ARFF format.

    Args:
        df: DataFrame with features and class column.
        output_path: Path for the .arff file.
        feature_columns: List of feature column names. If None, auto-detect
                         numeric columns (excluding class_column).
        class_column: Name of the class label column.
    """
    if feature_columns is None:
        feature_columns = [c for c in df.columns
                           if c != class_column and df[c].dtype in ("float64", "float32", "int64")]

    classes = sorted(df[class_column].dropna().unique())
    classes_str = ",".join(str(int(c)) for c in classes)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"@RELATION MVR_EEG_Features\n\n")

        for col in feature_columns:
            f.write(f"@ATTRIBUTE {col} NUMERIC\n")

        f.write(f"@ATTRIBUTE class {{{classes_str}}}\n\n")
        f.write("@DATA\n")

        for _, row in df.iterrows():
            values = []
            for col in feature_columns:
                val = row[col]
                if pd.isna(val):
                    values.append("?")
                else:
                    values.append(f"{val:.6f}")
            values.append(str(int(row[class_column])))
            f.write(",".join(values) + "\n")


def prepare_weka_dataset(df: pd.DataFrame, output_dir: str = None,
                          subset: str = None):
    """Prepare WEKA-ready CSV and ARFF files from a features DataFrame.

    Args:
        df: Full features DataFrame.
        output_dir: Output directory (default: WEKA_DIR).
        subset: Feature subset key from config (default: "DELTA").
    """
    output_dir = output_dir or cfg.WEKA_DIR
    subset = subset or cfg.WEKA_DEFAULT_SUBSET

    feature_cols = cfg.WEKA_FEATURE_SUBSETS.get(subset)
    if feature_cols is None:
        raise ValueError(f"Unknown feature subset: {subset}. Options: {list(cfg.WEKA_FEATURE_SUBSETS.keys())}")

    # Filter to columns that exist
    existing_cols = [c for c in feature_cols if c in df.columns]
    df_weka = df[existing_cols + ["MVR_Class"]].dropna()
    df_weka = df_weka.rename(columns={"MVR_Class": "class"})

    # Per-paradigm splits
    if "Paradigm" in df.columns:
        for paradigm in df["Paradigm"].unique():
            if pd.isna(paradigm):
                continue
            sub = df[df["Paradigm"] == paradigm]
            sub_cols = [c for c in existing_cols if c in sub.columns]
            sub_clean = sub[sub_cols + ["MVR_Class"]].dropna()
            sub_clean = sub_clean.rename(columns={"MVR_Class": "class"})

            base = f"Features_{paradigm}_{subset}"
            sub_clean.to_csv(os.path.join(output_dir, f"{base}.csv"), index=False)
            export_arff(sub_clean, os.path.join(output_dir, f"{base}.arff"),
                        feature_columns=sub_cols, class_column="class")

    # Full dataset
    base = f"Features_ALL_{subset}"
    df_weka.to_csv(os.path.join(output_dir, f"{base}.csv"), index=False)
    export_arff(df_weka, os.path.join(output_dir, f"{base}.arff"),
                feature_columns=existing_cols, class_column="class")

    return output_dir
