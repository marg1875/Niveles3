"""Manual hyperparameter tuning for best fractal configuration.
Evaluates: Basic per-channel (RS, Higuchi, DFA, Variogram x 16 ch = 64 feat)
3-class with SMOTE, Mes 3 and Mes 6.
Tracks both Pooled Accuracy and Mean-per-Patient Accuracy.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from itertools import product

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

ALL_CHANNELS = cfg.CHANNEL_NAMES
PER_CH_METHODS_BASIC = ["RS", "Higuchi", "DFA", "Variogram"]

# Only best months for 3-class
MONTHS_TUNE = [("Mes 3", "Month3"), ("Mes 6", "Month6")]

# Parameter grids (~10-20 combos per model)
PARAM_GRIDS = {
    "LogisticRegression": {
        "model_fn": lambda **kw: LogisticRegression(max_iter=5000, random_state=42, **kw),
        "grid": {
            "C": [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
            "solver": ["lbfgs", "newton-cholesky", "saga"],
        },
        "default": {"C": 1.0, "solver": "lbfgs"},
    },
    "SVM": {
        "model_fn": lambda **kw: SVC(random_state=42, probability=True, **kw),
        "grid": {
            "C": [0.1, 1, 10, 100],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto", 0.01, 0.1],
        },
        "default": {"C": 1.0, "kernel": "rbf", "gamma": "scale"},
    },
    "kNN": {
        "model_fn": lambda **kw: KNeighborsClassifier(n_jobs=-1, **kw),
        "grid": {
            "n_neighbors": [3, 5, 7, 10, 15, 20],
            "metric": ["chebyshev", "euclidean", "manhattan"],
            "weights": ["uniform", "distance"],
        },
        "default": {"n_neighbors": 10, "metric": "chebyshev", "weights": "uniform"},
    },
    "RandomForest": {
        "model_fn": lambda **kw: RandomForestClassifier(random_state=42, n_jobs=-1, **kw),
        "grid": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "max_features": ["sqrt", "log2", None],
        },
        "default": {"n_estimators": 300, "max_depth": None, "min_samples_split": 2, "max_features": "sqrt"},
    },
    "NaiveBayes": {
        "model_fn": lambda **kw: GaussianNB(**kw),
        "grid": {
            "var_smoothing": [1e-10, 1e-9, 1e-8, 1e-7, 1e-6],
        },
        "default": {"var_smoothing": 1e-9},
    },
    "MLP": {
        "model_fn": lambda **kw: MLPClassifier(max_iter=2000, early_stopping=True, random_state=42, **kw),
        "grid": {
            "hidden_layer_sizes": [(7,), (14,), (28,), (7, 7), (14, 7)],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate_init": [0.001, 0.01, 0.1],
        },
        "default": {"hidden_layer_sizes": (7,), "alpha": 0.0001, "learning_rate_init": 0.001},
    },
    "DecisionTree": {
        "model_fn": lambda **kw: DecisionTreeClassifier(random_state=42, **kw),
        "grid": {
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10, 20],
            "max_features": ["sqrt", "log2", None],
            "criterion": ["gini", "entropy"],
        },
        "default": {"max_depth": None, "min_samples_split": 2, "max_features": None, "criterion": "gini"},
    },
}


def build_features(df_raw):
    cols = []
    for ch in ALL_CHANNELS:
        for m in PER_CH_METHODS_BASIC:
            col = f"Active_{ch}_{m}"
            if col in df_raw.columns:
                cols.append(col)
    return cols


def evaluate_model(X_c, y_c, pat, model_fn, params, use_smote=True):
    """Run intra-subject CV with given params. Returns (pool_acc, pat_mean, pat_std)."""
    model = model_fn(**params)
    all_yt, all_yp = [], []
    patient_accs = []
    sm = SMOTE(random_state=42) if use_smote else None

    for patient in sorted(set(pat)):
        p_mask = pat == patient
        X_p, y_p = X_c[p_mask], y_c[p_mask]
        if len(y_p) < 10 or len(np.unique(y_p)) < 2:
            continue
        n_splits = min(5, max(2, len(y_p) // 3))
        try:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        except Exception:
            continue
        patient_yt, patient_yp = [], []
        for tr, te in skf.split(X_p, y_p):
            X_tr, X_te = X_p[tr], X_p[te]
            y_tr, y_te = y_p[tr], y_p[te]
            if sm is not None:
                try:
                    X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
                except Exception:
                    pass
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            model.fit(X_tr_s, y_tr)
            y_pred = model.predict(X_te_s)
            patient_yt.extend(y_te)
            patient_yp.extend(y_pred)
            all_yt.extend(y_te)
            all_yp.extend(y_pred)
        if patient_yt:
            patient_accs.append(accuracy_score(patient_yt, patient_yp))

    if len(all_yt) == 0:
        return np.nan, np.nan, np.nan
    pool_acc = accuracy_score(all_yt, all_yp)
    pat_mean = np.mean(patient_accs) if patient_accs else np.nan
    pat_std = np.std(patient_accs) if patient_accs else np.nan
    return pool_acc, pat_mean, pat_std


def params_to_str(params):
    return ", ".join(f"{k}={v}" for k, v in params.items())


def main():
    csv_path = os.path.join(cfg.FEATURES_DIR, "Fixed_W700_3class_Features.csv")
    df_raw = pd.read_csv(csv_path)
    feat_cols = build_features(df_raw)
    print(f"Features: {len(feat_cols)} ({len(PER_CH_METHODS_BASIC)} methods x {len(ALL_CHANNELS)} channels)")
    print(f"Models: {len(PARAM_GRIDS)}")
    print(f"Months: {[m[0] for m in MONTHS_TUNE]}")
    print(f"Total trials: ~{sum(len(list(product(*g['grid'].values()))) for g in PARAM_GRIDS.values()) * len(MONTHS_TUNE)}")
    print("=" * 100)

    t0 = time.time()
    all_results = []

    for month_label, month_val in MONTHS_TUNE:
        print(f"\n{'='*100}")
        print(f"  {month_label.upper()} (3-class, SMOTE, Basic per-channel 64 feat)")
        print(f"{'='*100}")

        df_f = df_raw.copy()
        if month_val:
            df_f = df_f[df_f["Session"] == month_val].copy()
        y = df_f["MVR_Class"].values
        X = df_f[feat_cols].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        pat = df_f["Patient"].values
        print(f"  Epochs: {len(y)}, Patients: {sorted(set(pat))}")

        for model_name in ["LogisticRegression", "SVM", "kNN", "RandomForest",
                            "NaiveBayes", "MLP", "DecisionTree"]:
            grid_info = PARAM_GRIDS[model_name]
            grid = grid_info["grid"]
            defaults = grid_info["default"]
            model_fn = grid_info["model_fn"]

            # First, evaluate defaults for baseline
            baseline_pool, baseline_pat, baseline_std = evaluate_model(
                X, y, pat, model_fn, defaults, use_smote=True)
            print(f"\n  {model_name} (default): Pool={baseline_pool*100:.2f}%  "
                  f"PatMean={baseline_pat*100:.2f}%+-{baseline_std*100:.1f}")

            # Generate all parameter combinations
            keys = list(grid.keys())
            values = list(grid.values())
            combos = list(product(*values))

            # If too many combos, sample or reduce
            MAX_COMBOS = 50
            if len(combos) > MAX_COMBOS:
                np.random.seed(42)
                idx = np.random.choice(len(combos), MAX_COMBOS, replace=False)
                combos = [combos[i] for i in idx]
            total = len(combos)

            best_pool = {"params": None, "acc": 0}
            best_pat = {"params": None, "acc": 0}

            for i, combo in enumerate(combos):
                params = dict(zip(keys, combo))
                pool_acc, pat_mean, pat_std = evaluate_model(
                    X, y, pat, model_fn, params, use_smote=True)

                all_results.append({
                    "Month": month_label, "Model": model_name,
                    "Params": params_to_str(params),
                    "Pooled_Accuracy": pool_acc,
                    "Patient_Accuracy_Mean": pat_mean,
                    "Patient_Accuracy_Std": pat_std,
                    "Pooled_Baseline": baseline_pool,
                    "Patient_Baseline": baseline_pat,
                })

                if pool_acc > best_pool["acc"]:
                    best_pool = {"params": params, "acc": pool_acc}
                if pat_mean > best_pat["acc"]:
                    best_pat = {"params": params, "acc": pat_mean, "std": pat_std}

                if (i + 1) % max(1, total // 4) == 0 or i == total - 1:
                    elapsed = time.time() - t0
                    print(f"    [{i+1:3d}/{total}] BestPool={best_pool['acc']*100:.2f}%  "
                          f"BestPat={best_pat['acc']*100:.2f}%+-{best_pat.get('std',0)*100:.1f}  "
                          f"({elapsed:.0f}s)")

            # Print best vs baseline
            delta_pool = best_pool["acc"] - baseline_pool
            delta_pat = best_pat["acc"] - baseline_pat
            print(f"  => Best Pool:  {best_pool['acc']*100:.2f}% (+{delta_pool*100:.1f}) "
                  f"{params_to_str(best_pool['params'])}")
            print(f"  => Best PatMean: {best_pat['acc']*100:.2f}%+-{best_pat['std']*100:.1f} (+{delta_pat*100:.1f}) "
                  f"{params_to_str(best_pat['params'])}")

    elapsed = time.time() - t0
    print(f"\n{'='*100}")
    print(f"Done in {elapsed/60:.1f} min. {len(all_results)} trials.")

    df_r = pd.DataFrame(all_results)
    out_path = os.path.join(cfg.CLASSIFICATION_DIR, "results_hyperparameter_tuning.csv")
    df_r.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # Summary table
    print(f"\n{'='*100}")
    print("SUMMARY: BEST PARAMS PER MODEL PER MONTH (3-class, Pooled Accuracy)")
    print(f"{'='*100}")
    for month_label, _ in MONTHS_TUNE:
        print(f"\n--- {month_label} ---")
        print(f"  {'Model':<20s} {'Default Pool':>10s} {'Best Pool':>10s} {'Delta':>7s} {'Best PatMean':>12s} {'Best Params'}")
        for model_name in ["LogisticRegression", "SVM", "kNN", "RandomForest",
                            "NaiveBayes", "MLP", "DecisionTree"]:
            ms = df_r[(df_r.Month == month_label) & (df_r.Model == model_name)]
            if len(ms) == 0:
                continue
            best = ms.loc[ms.Pooled_Accuracy.idxmax()]
            base_pool = best.Pooled_Baseline
            best_pool = best.Pooled_Accuracy
            delta = best_pool - base_pool
            best_pat = best.Patient_Accuracy_Mean
            print(f"  {model_name:<20s} {base_pool*100:>9.2f}% {best_pool*100:>9.2f}% {delta*100:>+6.1f}% "
                  f"{best_pat*100:>10.2f}%+-{best.Patient_Accuracy_Std*100:.1f}  {best.Params}")

    print(f"\n{'='*100}")
    print("SUMMARY: BEST PARAMS PER MODEL PER MONTH (Patient_Accuracy_Mean)")
    print(f"{'='*100}")
    for month_label, _ in MONTHS_TUNE:
        print(f"\n--- {month_label} ---")
        print(f"  {'Model':<20s} {'Default Pat':>10s} {'Best PatMean':>12s} {'Delta':>7s} {'Best Params'}")
        for model_name in ["LogisticRegression", "SVM", "kNN", "RandomForest",
                            "NaiveBayes", "MLP", "DecisionTree"]:
            ms = df_r[(df_r.Month == month_label) & (df_r.Model == model_name)]
            if len(ms) == 0:
                continue
            base_pat = ms.iloc[0].Patient_Baseline
            best = ms.loc[ms.Patient_Accuracy_Mean.idxmax()]
            best_pat = best.Patient_Accuracy_Mean
            delta = best_pat - base_pat
            print(f"  {model_name:<20s} {base_pat*100:>9.2f}% {best_pat*100:>10.2f}%+-{best.Patient_Accuracy_Std*100:.1f} {delta*100:>+6.1f}% "
                  f"{best.Params}")


if __name__ == "__main__":
    main()
