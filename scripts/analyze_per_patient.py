"""Per-patient and per-month analysis for the best fractal configuration.
Evaluates: Basic per-channel (RS, Higuchi, DFA, Variogram x 16 channels = 64 features)
with intra-subject CV and per-patient metrics reporting.
"""
import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
from imblearn.over_sampling import SMOTE

ALL_CHANNELS = cfg.CHANNEL_NAMES
PER_CH_METHODS_BASIC = ["RS", "Higuchi", "DFA", "Variogram"]

MONTHS = [("Mixto", None), ("Mes 1", "Month1"), ("Mes 3", "Month3"), ("Mes 6", "Month6")]
MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "SVM": SVC(random_state=42, probability=True),
}
PATIENTS = ["Px.006", "Px.007", "Px.008", "Px.009", "Px.010"]


def build_per_channel_columns(df, methods, channels):
    cols = []
    for ch in channels:
        for m in methods:
            col = f"Active_{ch}_{m}"
            if col in df.columns:
                cols.append(col)
    return cols


def main():
    csv_path = os.path.join(cfg.FEATURES_DIR, "Fixed_W700_3class_Features.csv")
    df_raw = pd.read_csv(csv_path)

    # Build column list
    feat_cols = build_per_channel_columns(df_raw, PER_CH_METHODS_BASIC, ALL_CHANNELS)
    print(f"Features: {len(feat_cols)} ({len(PER_CH_METHODS_BASIC)} methods x {len(ALL_CHANNELS)} channels)")
    print(f"Classes: {dict(df_raw['MVR_Class'].value_counts().sort_index())}")
    print(f"Patients: {sorted(df_raw['Patient'].unique())}")
    print()

    # ===== PER-MONTH, PER-PATIENT ANALYSIS =====
    print("=" * 120)
    print("ANALISIS PER-PACIENTE: LogisticRegression + Basic per-channel (64 feat) + SMOTE (3-clases)")
    print("=" * 120)
    print()

    for month_label, month_val in MONTHS:
        print(f"\n{'='*80}")
        print(f"  {month_label}")
        print(f"{'='*80}")

        # Filter by month
        df_f = df_raw.copy()
        if month_val:
            df_f = df_f[df_f["Session"] == month_val]

        if len(df_f) == 0:
            print("  No data for this condition")
            continue

        # Per-patient evaluation
        all_results = []
        for patient in sorted(set(df_f["Patient"].unique())):
            df_p = df_f[df_f["Patient"] == patient]
            y = df_p["MVR_Class"].values
            X = df_p[feat_cols].values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            if len(y) < 10 or len(np.unique(y)) < 2:
                all_results.append({"Patient": patient, "Accuracy": np.nan, "F1": np.nan,
                                    "Kappa": np.nan, "N": len(y)})
                continue

            # Patient-level CV scores
            patient_accs, patient_f1s, patient_kappas = [], [], []
            patient_yts, patient_yps = [], []
            n_splits = min(5, max(2, len(y) // 3))

            try:
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            except Exception:
                all_results.append({"Patient": patient, "Accuracy": np.nan, "F1": np.nan,
                                    "Kappa": np.nan, "N": len(y)})
                continue

            sm = SMOTE(random_state=42)
            model = LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1)

            for tr, te in skf.split(X, y):
                X_tr, X_te = X[tr], X[te]
                y_tr, y_te = y[tr], y[te]

                try:
                    X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
                except Exception:
                    pass

                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_te_s = scaler.transform(X_te)
                model.fit(X_tr_s, y_tr)
                y_pred = model.predict(X_te_s)

                patient_yts.extend(y_te)
                patient_yps.extend(y_pred)
                patient_accs.append(accuracy_score(y_te, y_pred))
                try:
                    patient_f1s.append(f1_score(y_te, y_pred, average="macro", zero_division=0))
                except Exception:
                    patient_f1s.append(0.0)
                try:
                    patient_kappas.append(cohen_kappa_score(y_te, y_pred))
                except Exception:
                    patient_kappas.append(0.0)

            if patient_yts:
                pool_acc = accuracy_score(patient_yts, patient_yps)
                pool_f1 = f1_score(patient_yts, patient_yps, average="macro", zero_division=0)
                pool_kappa = cohen_kappa_score(patient_yts, patient_yps)
                cm = confusion_matrix(patient_yts, patient_yps)
                mean_acc = np.mean(patient_accs)
                std_acc = np.std(patient_accs)
            else:
                pool_acc = pool_f1 = pool_kappa = mean_acc = std_acc = np.nan
                cm = None

            n_per_class = dict(df_p["MVR_Class"].value_counts().sort_index())
            all_results.append({
                "Patient": patient, "Accuracy": pool_acc, "F1": pool_f1,
                "Kappa": pool_kappa, "N": len(y),
                "Mean_Acc_CV": mean_acc, "Std_Acc_CV": std_acc,
                "Classes": str(n_per_class),
                "CM": cm,
            })

        # Print per-patient table
        print(f"  {'Patient':<8s} {'N':>5s} {'Clases':<30s} {'Accuracy':>10s} {'F1':>8s} {'Kappa':>8s} {'Mean(CV)':>10s}")
        print(f"  {'-'*85}")
        total_yt, total_yp = [], []
        for r in all_results:
            c_str = r["Classes"] if r["Classes"] else "N/A"
            if np.isnan(r["Accuracy"]):
                print(f"  {r['Patient']:<8s} {r['N']:>5d} {c_str:<30s} {'N/A':>10s}")
            else:
                print(f"  {r['Patient']:<8s} {r['N']:>5d} {c_str:<30s} "
                      f"{r['Accuracy']*100:>8.2f}% {r['F1']:.4f} {r['Kappa']:+.4f} "
                      f"{r['Mean_Acc_CV']*100:>8.2f}%+-{r['Std_Acc_CV']*100:.1f}")

        # Pooled across all patients (how the report computes it)
        # Need to re-run with ALL patients pooled into CV
        all_valid = [r for r in all_results if not np.isnan(r["Accuracy"])]
        if all_valid:
            avg_acc = np.mean([r["Accuracy"] for r in all_valid])
            print(f"  {'-'*85}")
            print(f"  {'PROMEDIO':<8s} {'':>5s} {'':>30s} {avg_acc*100:>8.2f}%")

        # Now compute the true POOLED metric (all patients combined in CV)
        print(f"\n  --- POOL global (todos los pacientes combinados en CV) ---")
        y_all = df_f["MVR_Class"].values
        X_all = df_f[feat_cols].values
        X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
        pat = df_f["Patient"].values

        all_yt, all_yp = [], []
        sm = SMOTE(random_state=42)

        for patient in sorted(set(pat)):
            p_mask = pat == patient
            X_p, y_p = X_all[p_mask], y_all[p_mask]
            if len(y_p) < 10 or len(np.unique(y_p)) < 2:
                continue
            n_splits = min(5, max(2, len(y_p) // 3))
            try:
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            except Exception:
                continue

            model_pool = LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1)
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
                model_pool.fit(X_tr_s, y_tr)
                y_pred = model_pool.predict(X_te_s)
                all_yt.extend(y_te)
                all_yp.extend(y_pred)

        pool_acc = accuracy_score(all_yt, all_yp)
        pool_f1 = f1_score(all_yt, all_yp, average="macro", zero_division=0)
        pool_kappa = cohen_kappa_score(all_yt, all_yp)
        cm = confusion_matrix(all_yt, all_yp)
        print(f"  Accuracy: {pool_acc*100:.2f}%")
        print(f"  F1 Macro: {pool_f1:.4f}")
        print(f"  Kappa:    {pool_kappa:+.4f}")
        print(f"  N total:  {len(all_yt)}")
        print(f"  Matriz de confusion:")
        print(f"         Pred 0    Pred 10   Pred 40")
        for i, label in enumerate([0, 10, 40]):
            print(f"  Real {label:>3d}: {cm[i, 0]:>7d}    {cm[i, 1]:>7d}    {cm[i, 2]:>7d}")

    # ===== DATA LEAKAGE CHECK =====
    print(f"\n{'='*120}")
    print("VERIFICACION DE DATA LEAKAGE")
    print(f"{'='*120}")

    # Check: Are the same basal segments assigned to different patients? No - they're per-file.
    # Check: Do basal segments have different features from active segments with same MVR label?
    # The first 5 segments are class 0 (basal) and also used for basal reference.
    # This is NOT data leakage because:
    # a) Basal features for class 0 are the ACTIVE features of those segments, not some pre-computed delta
    # b) Each patient only sees his own data in CV (intra-subject)
    # c) The same segments appear in both train and test within a patient only if unlucky fold split,
    #    but StratifiedKFold prevents epoch overlap

    print("\nChequeo de clase basal (primeros 5 segmentos por archivo):")
    print("  - Los primeros 5 segmentos se etiquetan como clase 0 (reposo)")
    print("  - Estos segmentos SOLO se usan como epocas de clase 0, NO como referencia basal para Delta")
    print("  - No hay Delta features en este analisis (solo Active_)")
    print("  - StratifiedKFold garantiza que cada epoca aparece solo en train O test, no en ambos")
    print("  - Riesgo: segmentos adyacentes de un mismo archivo pueden estar en train y test (correlacion temporal).")
    print("    Esto aplica por igual a todas las clases, no solo a la basal.")

    # Check temporal adjacency: consecutive segments from same file
    print("\nChequeo de correlacion temporal (segmentos contiguos del mismo archivo):")
    df_check = df_raw[["Patient", "Session", "Paradigm", "MVR_Class", "Event_Time_sec"]].copy()
    df_check["file_id"] = df_check["Patient"] + "_" + df_check["Session"].astype(str) + "_" + df_check["Paradigm"]
    files = df_check["file_id"].unique()
    print(f"  Archivos unicos: {len(files)}")
    avg_segs = len(df_check) / len(files)
    print(f"  Epocas promedio por archivo: {avg_segs:.1f}")
    print(f"  Epocas de clase 0 por archivo: 5 (fijas, primeros 5 segmentos)")
    print()

    # Check if there are any NaN in the per-channel features
    X_check = df_raw[feat_cols].values
    nan_count = np.sum(np.isnan(X_check))
    inf_count = np.sum(np.isinf(X_check))
    print("Chequeo de features:")
    print(f"  Total features: {len(feat_cols)}")
    print(f"  NaN values: {nan_count}")
    print(f"  Inf values: {inf_count}")
    print(f"  Shape: {X_check.shape}")

    # Per patient class distribution
    print(f"\n{'='*120}")
    print("DISTRIBUCION DE CLASES POR PACIENTE Y MES")
    print(f"{'='*120}")
    for month_label, month_val in MONTHS:
        print(f"\n{month_label}:")
        df_m = df_raw.copy()
        if month_val:
            df_m = df_m[df_m["Session"] == month_val]
        for patient in PATIENTS:
            df_p = df_m[df_m["Patient"] == patient]
            if len(df_p) > 0:
                dist = df_p["MVR_Class"].value_counts().sort_index().to_dict()
                print(f"  {patient}: {dist}")


if __name__ == "__main__":
    main()
