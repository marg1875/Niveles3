"""Fine hyperparameter tuning around SVM and LogReg optima.
SVM: C=[5,7,10,12,15,20], gamma around scale, poly kernel, class_weight
LogReg: L1/L2/ElasticNet penalty, fine C, l1_ratio
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

ALL_CHANNELS = cfg.CHANNEL_NAMES
PER_CH_METHODS_BASIC = ["RS", "Higuchi", "DFA", "Variogram"]

PARAM_GRIDS_FINE = {
    "SVM": {
        "model_fn": lambda **kw: SVC(random_state=42, probability=True, cache_size=2000, **kw),
        "grids": [
            # Grid 1: rbf kernel, fine C + gamma around optimum
            {
                "C": [5, 7, 10, 12, 15, 20],
                "kernel": ["rbf"],
                "gamma": [0.001, 0.005, 0.01, 0.02, 0.05, "scale", "auto"],
                "class_weight": [None, "balanced"],
            },
            # Grid 2: poly kernel (untested)
            {
                "C": [1, 10, 100],
                "kernel": ["poly"],
                "degree": [2, 3, 4],
                "gamma": ["scale", "auto", 0.01],
                "coef0": [0.0, 1.0],
                "class_weight": [None],
            },
        ],
        "best_so_far": {"C": 10, "kernel": "rbf", "gamma": "scale", "class_weight": None},
    },
    "LogisticRegression": {
        "model_fn": lambda **kw: LogisticRegression(max_iter=5000, random_state=42, **kw),
        "grids": [
            # Grid 1: L2 with fine C
            {
                "C": [2, 3, 5, 7, 10, 15],
                "penalty": ["l2"],
                "solver": ["lbfgs"],
                "class_weight": [None, "balanced"],
            },
            # Grid 2: L1/L2/ElasticNet with saga solver
            {
                "C": [1, 3, 5, 10],
                "penalty": ["l1", "l2", "elasticnet"],
                "solver": ["saga"],
                "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
                "class_weight": [None, "balanced"],
            },
        ],
        "best_so_far": {"C": 5, "solver": "saga", "penalty": "l2", "class_weight": None},
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


def evaluate_model(X_c, y_c, pat, model_fn, params):
    model = model_fn(**params)
    all_yt, all_yp = [], []
    patient_accs = []
    sm = SMOTE(random_state=42)

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
    print(f"Fine-tuning: SVM + LogReg, Mes 3 + Mes 6, 3-class + SMOTE")

    t0 = time.time()
    all_results = []
    MONTHS = [("Mes 3", "Month3"), ("Mes 6", "Month6")]

    for month_label, month_val in MONTHS:
        df_f = df_raw.copy()
        if month_val:
            df_f = df_f[df_f["Session"] == month_val].copy()
        y = df_f["MVR_Class"].values
        X = df_f[feat_cols].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        pat = df_f["Patient"].values

        for model_name in ["SVM", "LogisticRegression"]:
            info = PARAM_GRIDS_FINE[model_name]
            best_prev = info["best_so_far"]
            baseline_pool, baseline_pat, baseline_std = evaluate_model(
                X, y, pat, info["model_fn"], best_prev)

            print(f"\n{'='*80}")
            print(f"  {month_label} - {model_name}")
            print(f"  Best so far: {baseline_pool*100:.2f}% (Pool) | {baseline_pat*100:.2f}%+-{baseline_std*100:.1f} (PatMean)")
            print(f"  Params: {params_to_str(best_prev)}")

            best_pool = {"params": best_prev.copy(), "acc": baseline_pool}
            best_pool_acc = baseline_pool
            best_pat = {"params": best_prev.copy(), "acc": baseline_pat, "std": baseline_std}
            best_pat_acc = baseline_pat

            total_combos = 0
            for gi, grid in enumerate(info["grids"]):
                keys = list(grid.keys())
                vals = list(grid.values())
                combos = list(product(*vals))
                total_combos += len(combos)

                for i, combo in enumerate(combos):
                    params = dict(zip(keys, combo))

                    # Skip meaningless combos: l1_ratio only for elasticnet
                    if "l1_ratio" in params and "penalty" in params and params["penalty"] != "elasticnet":
                        continue
                    # Class weight + SMOTE might be redundant but worth testing
                    # Skip poly with class_weight (too many combos)
                    if params.get("kernel") == "poly" and params.get("class_weight") == "balanced":
                        continue

                    pool_acc, pat_mean, pat_std = evaluate_model(
                        X, y, pat, info["model_fn"], params)

                    all_results.append({
                        "Month": month_label, "Model": model_name,
                        "Params": params_to_str(params),
                        "Pooled_Accuracy": pool_acc,
                        "Patient_Accuracy_Mean": pat_mean,
                        "Patient_Accuracy_Std": pat_std,
                        "Best_So_Far_Pool": baseline_pool,
                        "Best_So_Far_Pat": baseline_pat,
                        "Grid": gi + 1,
                    })

                    if pool_acc > best_pool_acc:
                        best_pool = {"params": params, "acc": pool_acc}
                        best_pool_acc = pool_acc
                        print(f"    [!] NEW BEST Pool:  {pool_acc*100:.2f}% ({params_to_str(params)})")
                    if pat_mean > best_pat_acc:
                        best_pat = {"params": params, "acc": pat_mean, "std": pat_std}
                        best_pat_acc = pat_mean

                    if (i + 1) % max(1, len(combos) // 3) == 0:
                        elapsed = time.time() - t0
                        print(f"    [{i+1:3d}/{len(combos):3d}] Grid {gi+1} | BestPool={best_pool_acc*100:.2f}% BestPat={best_pat_acc*100:.2f}%+-{best_pat.get('std',0)*100:.1f} ({elapsed:.0f}s)")

            # Baseline re-evaluation for consistency
            print(f"  => FINAL Best Pool:  {best_pool_acc*100:.2f}% (params: {params_to_str(best_pool['params'])})")
            print(f"  => FINAL Best PatMean: {best_pat_acc*100:.2f}%+-{best_pat['std']*100:.1f} (params: {params_to_str(best_pat['params'])})")
            print(f"  Delta vs previous best: Pool={(best_pool_acc - baseline_pool)*100:+.2f} pts, PatMean={(best_pat_acc - baseline_pat)*100:+.2f} pts")

            # Periodic save
            pd.DataFrame(all_results).to_csv(
                os.path.join(cfg.CLASSIFICATION_DIR, f"results_finetuning_svm_logreg_partial.csv"),
                index=False)

    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"Done in {elapsed/60:.1f} min. {len(all_results)} trials.")

    df_r = pd.DataFrame(all_results)
    out_path = os.path.join(cfg.CLASSIFICATION_DIR, "results_finetuning_svm_logreg.csv")
    df_r.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY: Pre-Tuning vs Post-Tuning (3-class, Pooled Accuracy)")
    print(f"{'='*80}")
    print(f"  {'Model':<20s} {'Month':<8s} {'Pre-Tuning':>11s} {'Fine-Tuned':>11s} {'Delta':>7s} {'Best Params'}")
    print(f"  {'-'*100}")
    for month_label in [m[0] for m in MONTHS]:
        for model_name in ["SVM", "LogisticRegression"]:
            ms = df_r[(df_r.Month == month_label) & (df_r.Model == model_name)]
            if len(ms) == 0:
                continue
            best = ms.loc[ms.Pooled_Accuracy.idxmax()]
            baseline = best.Best_So_Far_Pool
            tuned = best.Pooled_Accuracy
            delta = tuned - baseline
            print(f"  {model_name:<20s} {month_label:<8s} {baseline*100:>10.2f}% {tuned*100:>10.2f}% {delta*100:>+6.2f}%  {best.Params}")


if __name__ == "__main__":
    main()
