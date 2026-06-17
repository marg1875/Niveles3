"""
ML classification evaluation for EEG fractal features.

Evaluates 9 algorithms across 5 channel configurations, 4 feature subsets,
and 2 classification types (3-class and 2-class 10vs40) using LOPO CV.

Output: results CSV, ROC data, comparison plots.
"""
import os
import sys
import time
import warnings
import itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")

import config as cfg
from src.classification.features import (
    get_channel_subset_features, get_feature_subset, filter_2class
)
from src.classification.validation import lopo_split
from src.classification.metrics import compute_metrics, aggregate_lopo_results

# Sklearn imports
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

# Model registry (Martinez-Peon 2024 + LDA + LogisticRegression)
MODELS = {
    "SVM": {
        "fn": lambda: SVC(random_state=42, probability=True),
        "params": {"C": [0.1, 1, 10], "kernel": ["rbf", "poly"], "degree": [3]},
    },
    "kNN": {
        "fn": lambda: KNeighborsClassifier(n_jobs=-1),
        "params": {"n_neighbors": [5, 10], "weights": ["uniform", "distance"],
                   "metric": ["euclidean", "chebyshev"]},
    },
    "MLP": {
        "fn": lambda: MLPClassifier(max_iter=1000, random_state=42, early_stopping=True),
        "params": {"hidden_layer_sizes": [(7,), (32,)], "learning_rate_init": [0.001, 0.3],
                   "alpha": [0.001, 0.01]},
    },
    "RandomForest": {
        "fn": lambda: RandomForestClassifier(random_state=42, n_jobs=-1),
        "params": {"n_estimators": [100, 300], "max_depth": [10, None]},
    },
    "NaiveBayes": {
        "fn": lambda: GaussianNB(),
        "params": {"var_smoothing": [1e-9, 1e-7]},
    },
    "BayesNet": {
        "fn": lambda: GaussianNB(),
        "params": {"var_smoothing": [1e-9, 1e-7, 1e-5]},
    },
    "RandomTree": {
        "fn": lambda: DecisionTreeClassifier(random_state=42),
        "params": {"max_features": ["sqrt", "log2"], "max_depth": [None, 10, 20]},
    },
    "LDA": {
        "fn": lambda: LinearDiscriminantAnalysis(),
        "params": {"solver": ["svd", "lsqr", "eigen"]},
    },
    "LogisticRegression": {
        "fn": lambda: LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1),
        "params": {"C": [0.01, 0.1, 1.0, 10.0]},
    },
}


def run_lopo_experiment(X, y, model_fn, param_grid, patient_col, df):
    """Run LOPO CV with inner grid search for a single model+feature combination.

    Args:
        X: Feature matrix.
        y: Labels.
        model_fn: Callable returning a sklearn classifier.
        param_grid: Dict of hyperparameter options for GridSearchCV.
        patient_col: List/Series of patient identifiers for LOPO splitting.
        df: Original DataFrame (for creating patient-based splits).

    Returns:
        Dict with aggregated metrics and ROC data.
    """
    classes = sorted(np.unique(y))
    fold_results = []
    y_true_all = []
    y_pred_all = []
    y_proba_all = []
    patient_df = pd.DataFrame({"Patient": patient_col})

    for train_idx, test_idx, test_patient in lopo_split(patient_df):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Skip if test patient has < 5 epochs
        if len(y_test) < 5:
            continue

        # Standardize features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Inner grid search on training fold
        if param_grid and len(param_grid) > 0:
            try:
                grid = GridSearchCV(
                    model_fn(), param_grid,
                    cv=min(2, len(np.unique(patient_df[train_idx]))),
                    scoring="f1_macro", n_jobs=-1, refit=True
                )
                grid.fit(X_train_s, y_train)
                best_model = grid.best_estimator_
            except Exception:
                best_model = model_fn()
                best_model.fit(X_train_s, y_train)
        else:
            best_model = model_fn()
            best_model.fit(X_train_s, y_train)

        # Predict and evaluate
        y_pred = best_model.predict(X_test_s)
        try:
            y_proba = best_model.predict_proba(X_test_s)
        except Exception:
            y_proba = None

        y_true_all.append(y_test)
        y_pred_all.append(y_pred)
        y_proba_all.append(y_proba)

        metrics = compute_metrics(y_test, y_pred, y_proba, classes=classes)
        fold_results.append(metrics)

    if not fold_results:
        return {"accuracy_mean": np.nan, "f1_macro_mean": np.nan}

    agg = aggregate_lopo_results(fold_results, classes)
    agg["roc_data"] = {
        "y_true": y_true_all,
        "y_pred": y_pred_all,
        "y_proba": y_proba_all,
    }
    return agg


def run_intra_experiment(X, y, model_fn, param_grid, patient_col):
    """Run intra-subject CV (StratifiedKFold per patient, pooled across patients).

    Comparable to Martinez-Peon 2024: 80/20 split within subjects.
    """
    classes = sorted(np.unique(y))
    fold_results = []
    y_true_all = []
    y_pred_all = []
    y_proba_all = []
    patient_df = pd.DataFrame({"Patient": patient_col})

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for patient in sorted(patient_df["Patient"].unique()):
        p_idx = (patient_df["Patient"] == patient).values
        X_p, y_p = X[p_idx], y[p_idx]
        n_p = len(y_p)

        if n_p < 10 or len(np.unique(y_p)) < 2:
            continue

        splitter = StratifiedKFold(n_splits=min(5, max(2, n_p // 5)), shuffle=True, random_state=42)
        for train_idx, test_idx in splitter.split(X_p, y_p):
            X_train, X_test = X_p[train_idx], X_p[test_idx]
            y_train, y_test = y_p[train_idx], y_p[test_idx]

            if len(y_test) < 3:
                continue

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            if param_grid and len(param_grid) > 0:
                try:
                    grid = GridSearchCV(model_fn(), param_grid, cv=min(3, len(y_train)//2),
                                        scoring="f1_macro", n_jobs=-1, refit=True)
                    grid.fit(X_train_s, y_train)
                    best_model = grid.best_estimator_
                except Exception:
                    best_model = model_fn()
                    best_model.fit(X_train_s, y_train)
            else:
                best_model = model_fn()
                best_model.fit(X_train_s, y_train)

            y_pred = best_model.predict(X_test_s)
            try:
                y_proba = best_model.predict_proba(X_test_s)
            except Exception:
                y_proba = None

            y_true_all.append(y_test)
            y_pred_all.append(y_pred)
            y_proba_all.append(y_proba)

            metrics = compute_metrics(y_test, y_pred, y_proba, classes=classes)
            fold_results.append(metrics)

    if not fold_results:
        return {"accuracy_mean": np.nan, "f1_macro_mean": np.nan}

    agg = aggregate_lopo_results(fold_results, classes)
    agg["roc_data"] = {
        "y_true": y_true_all,
        "y_pred": y_pred_all,
        "y_proba": y_proba_all,
    }
    return agg


def main():
    print("=" * 60)
    print("NIVELES3 — ML CLASSIFICATION EVALUATION")
    print("=" * 60)
    print(f"Validation: Leave-One-Patient-Out (LOPO)")
    print(f"Models: {len(MODELS)} ({', '.join(MODELS.keys())})")
    print(f"Channel configs: 5")
    print(f"Feature subsets: 4")
    print("=" * 60)
    print()

    # Load data
    csv_path = os.path.join(cfg.FEATURES_DIR, "Global_All_Features.csv")
    print(f"Loading: {csv_path}")
    df_raw = pd.read_csv(csv_path)
    print(f"Total epochs: {len(df_raw)}, Patients: {sorted(df_raw['Patient'].unique())}")
    print()

    # Experiment combinations
    channel_configs = ["Cz", "Motor-2", "Motor-4", "Motor-6", "All-16"]
    feature_subsets = ["Delta", "Martinez", "Martinez-All-p", "Delta+Martinez", "Delta+Spectral"]
    class_types = ["3class", "2class"]
    validation_types = ["LOPO", "IntraSubject"]

    total_experiments = (len(MODELS) * len(channel_configs) *
                         len(feature_subsets) * len(class_types) * len(validation_types))
    print(f"Total experiments: {total_experiments}")
    print()

    # Ensure output directories exist
    os.makedirs(cfg.ROC_DATA_DIR, exist_ok=True)

    results = []
    experiment_count = 0
    t0 = time.time()

    pbar = tqdm(total=total_experiments, desc="Experiments", unit="exp")

    for ch_conf in channel_configs:
        # Get channel-averaged features
        df_ch = get_channel_subset_features(df_raw, ch_conf)

        for feat_subset in feature_subsets:
            try:
                X_full, y_full, feat_cols, df_filt = get_feature_subset(df_ch, feat_subset)
            except Exception as e:
                print(f"\n  [SKIP] {ch_conf}/{feat_subset}: {e}")
                continue

            if len(feat_cols) < 2:
                print(f"\n  [SKIP] {ch_conf}/{feat_subset}: too few features ({len(feat_cols)})")
                continue

            for class_type in class_types:
                if class_type == "2class":
                    mask = np.isin(y_full, (10, 40))
                    X_c = X_full[mask]
                    y_c = y_full[mask]
                    pat_col_c = df_filt.iloc[mask]["Patient"].values
                else:
                    X_c = X_full
                    y_c = y_full
                    pat_col_c = df_filt["Patient"].values

                if len(y_c) < 10:
                    print(f"\n  [SKIP] {ch_conf}/{feat_subset}/{class_type}: too few samples ({len(y_c)})")
                    continue

                n_patients = len(np.unique(pat_col_c))
                if n_patients < 3:
                    print(f"\n  [SKIP] {ch_conf}/{feat_subset}/{class_type}: too few patients ({n_patients})")
                    continue

                for val_type in validation_types:
                    for model_name, model_def in MODELS.items():
                        experiment_count += 1
                        t_exp = time.time()

                        if val_type == "LOPO":
                            agg = run_lopo_experiment(
                                X_c, y_c, model_def["fn"], model_def["params"],
                                pat_col_c, df_raw
                            )
                        else:
                            agg = run_intra_experiment(
                                X_c, y_c, model_def["fn"], model_def["params"],
                                pat_col_c
                            )

                    exp_time = time.time() - t_exp

                    # Save ROC data (concatenate folds to handle unequal sizes)
                    roc_data = agg.pop("roc_data", None)
                    if roc_data is not None:
                        roc_path = os.path.join(
                            cfg.ROC_DATA_DIR,
                            f"{model_name}_{ch_conf}_{feat_subset}_{class_type}_{val_type}.npz"
                        )
                        classes = sorted(np.unique(y_c))
                        yt_flat = np.concatenate(roc_data["y_true"])
                        yp_flat = np.concatenate(roc_data["y_pred"])
                        yb_flat = np.concatenate([p for p in roc_data.get("y_proba", []) if p is not None]) if roc_data.get("y_proba") else None
                        if yb_flat is not None and len(yb_flat) == 0:
                            yb_flat = None
                        save_kw = {"y_true": yt_flat, "y_pred": yp_flat, "classes": classes}
                        if yb_flat is not None:
                            save_kw["y_proba"] = yb_flat
                        np.savez_compressed(roc_path, **save_kw)

                    row = {
                        "Model": model_name,
                        "Channel_Config": ch_conf,
                        "N_Channels": len(cfg.CHANNEL_SUBSETS[ch_conf]),
                        "Feature_Subset": feat_subset,
                        "N_Features": len(feat_cols),
                        "Class_Type": class_type,
                        "Validation": val_type,
                        "N_Samples": len(y_c),
                        "Time_sec": round(exp_time, 1),
                    }
                    row.update(agg)

                    # Primary metrics
                    acc = agg.get("accuracy_mean", np.nan)
                    f1 = agg.get("f1_macro_mean", np.nan)
                    n_nan = not np.isnan(acc)

                    results.append(row)

                    pbar.update(1)
                    if n_nan:
                        pbar.set_postfix_str(
                            f"[{experiment_count}/{total_experiments}] "
                            f"({100*experiment_count/total_experiments:.0f}%) "
                            f"{model_name} | {ch_conf} | {feat_subset} | {class_type} | Acc={acc:.3f}"
                        )
                    else:
                        pbar.set_postfix_str(
                            f"[{experiment_count}/{total_experiments}] "
                            f"({100*experiment_count/total_experiments:.0f}%) "
                            f"{model_name} | {ch_conf} | {feat_subset} | {class_type} | FAILED"
                        )

    pbar.close()

    # Save results
    results_df = pd.DataFrame(results)
    out_csv = os.path.join(cfg.CLASSIFICATION_DIR, "results_lopo.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\nResults saved: {out_csv}")

    # Summary
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Experiments: {experiment_count}/{total_experiments}")

    # Top results
    if not results_df.empty and "accuracy_mean" in results_df.columns:
        valid = results_df.dropna(subset=["accuracy_mean"])
        if not valid.empty:
            print("\n" + "=" * 60)
            print("TOP 10 RESULTS (by LOPO Accuracy)")
            print("=" * 60)
            top = valid.nlargest(10, "accuracy_mean")
            for _, r in top.iterrows():
                print(f"  {r['Model']:20s} | {r['Channel_Config']:16s} | "
                      f"{r['Feature_Subset']:14s} | {r['Class_Type']:6s} | "
                      f"Acc={r['accuracy_mean']:.3f} F1={r['f1_macro_mean']:.3f}")

            print("\n" + "=" * 60)
            print("BEST PER CHANNEL CONFIGURATION (3-class)")
            print("=" * 60)
            best_per_ch = valid[valid["Class_Type"] == "3class"].groupby("Channel_Config").apply(
                lambda g: g.nlargest(1, "accuracy_mean")
            ).reset_index(drop=True)
            for _, r in best_per_ch.iterrows():
                print(f"  {r['Channel_Config']:16s} ({r['N_Channels']}ch) -> "
                      f"{r['Model']:20s} | {r['Feature_Subset']:12s} | "
                      f"Acc={r['accuracy_mean']:.3f}")

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    sys.stderr = open(os.path.join(cfg.CLASSIFICATION_DIR, "warnings.log"), "w", encoding="utf-8")
    main()
