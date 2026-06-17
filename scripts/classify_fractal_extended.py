"""Extended fractal classification - Martinez-Peon 2024 style with SMOTE for 3-class."""
import os, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from src.classification.features import get_channel_subset_features, get_feature_subset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, mean_absolute_error
from imblearn.over_sampling import SMOTE

# ===== MODELS (7, matching Martinez-Peon 2024) =====
MODELS = {
    "SVM":                SVC(random_state=42, probability=True),
    "kNN":                KNeighborsClassifier(n_neighbors=10, metric="chebyshev", n_jobs=-1),
    "RandomForest":       RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "NaiveBayes":         GaussianNB(),
    "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1),
    "MLP":                MLPClassifier(hidden_layer_sizes=(7,), learning_rate_init=0.3,
                                         max_iter=1000, early_stopping=True, random_state=42),
    "DecisionTree":       DecisionTreeClassifier(max_features="sqrt", max_depth=10, random_state=42),
}

MODEL_ORDER = ["SVM", "kNN", "RandomForest", "NaiveBayes", "LogisticRegression", "MLP", "DecisionTree"]

# ===== CHANNEL CONFIGURATIONS =====
ALL_CHANNELS = cfg.CHANNEL_NAMES
CHANNEL_SUBSETS_CONFIG = {
    "Cz":          [ALL_CHANNELS[i] for i in cfg.CHANNEL_SUBSETS["Cz"]],
    "Motor-2":     [ALL_CHANNELS[i] for i in cfg.CHANNEL_SUBSETS["Motor-2"]],
    "Motor-4":     [ALL_CHANNELS[i] for i in cfg.CHANNEL_SUBSETS["Motor-4"]],
    "Motor-6":     [ALL_CHANNELS[i] for i in cfg.CHANNEL_SUBSETS["Motor-6"]],
    "Frontal-5":   [ALL_CHANNELS[i] for i in cfg.CHANNEL_SUBSETS["Frontal-5"]],
    "Parietal-5":  [ALL_CHANNELS[i] for i in cfg.CHANNEL_SUBSETS["Parietal-5"]],
    "All-16":      ALL_CHANNELS,
}

PER_CH_METHODS_BASIC = ["RS", "Higuchi", "DFA", "Variogram"]
PER_CH_METHODS_MARTINEZ = ["HO", "HRS_p64", "HV"]
PER_CH_METHODS_ALL = ["RS", "Higuchi", "DFA", "Variogram", "HO", "HRS_p64", "HV"]

# ===== HELPER FUNCTIONS =====

def _build_per_channel_columns(df, methods, channels):
    """Build per-channel column list: Active_{ch}_{method}."""
    cols = []
    for ch in channels:
        for m in methods:
            col = f"Active_{ch}_{m}"
            if col in df.columns:
                cols.append(col)
    return cols


def _extract_xy_spatial_mean(df_raw, exp, month_val):
    """Build X, y using spatial mean features (Groups A)."""
    df_ch = get_channel_subset_features(df_raw, "All-16")
    if month_val:
        df_ch = df_ch[df_ch["Session"] == month_val].copy()
    if "subset" in exp:
        X, y, feat_cols, df_filt = get_feature_subset(df_ch, exp["subset"])
    else:
        cols = [c for c in exp["columns"] if c in df_ch.columns]
        if not cols:
            raise ValueError(f"No columns found: {exp['columns']}")
        y = df_ch["MVR_Class"].values
        X = df_ch[cols].values
        feat_cols = cols
        df_filt = df_ch
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y, feat_cols, df_filt


def _extract_xy_per_channel(df_raw, exp, month_val):
    """Build X, y using per-channel features (Groups B, C)."""
    if exp["type"] == "pc":
        channels = ALL_CHANNELS
    else:
        ch_key = exp["channels"][0]
        channels = CHANNEL_SUBSETS_CONFIG[ch_key]
    cols = _build_per_channel_columns(df_raw, exp["methods"], channels)
    if not cols:
        raise ValueError(f"No per-channel columns for methods={exp['methods']} channels={channels}")
    df_f = df_raw.copy()
    if month_val:
        df_f = df_f[df_f["Session"] == month_val].copy()
    y = df_f["MVR_Class"].values
    X = df_f[cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y, cols, df_f


def _classify_patient_cv(X_c, y_c, pat, model, use_smote):
    """Intra-subject CV with optional SMOTE. Returns (y_true, y_pred) lists."""
    all_yt, all_yp = [], []
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
            all_yt.extend(y_te)
            all_yp.extend(y_pred)
    return all_yt, all_yp


# ===== EXPERIMENT DEFINITIONS =====

EXPERIMENTS = []

# --- Group A: Spatial Mean (9 subsets) ---
sm_individual = [
    ("A1_RS", "Rescaled Range", {"type": "sm", "subset": "RS_only"}),
    ("A2_Higuchi", "Higuchi", {"type": "sm", "subset": "Higuchi_only"}),
    ("A3_DFA", "DFA", {"type": "sm", "subset": "DFA_only"}),
    ("A4_Semivariograma", "Semivariograma", {"type": "sm", "subset": "Semivariograma_only"}),
    ("A5_Promedio", "Promedio", {"type": "sm_custom", "columns": ["Active_Average"]}),
    ("A6_HO", "Hurst Original", {"type": "sm", "subset": "HO_only"}),
    ("A7_HRS_p64", "HRS p64", {"type": "sm", "subset": "HRS_only"}),
    ("A8_HV", "Hurst Semivariograma", {"type": "sm", "subset": "HV_only"}),
]
for sid, label, params in sm_individual:
    EXPERIMENTS.append({"id": sid, "label": label, "group": "A", **params})

EXPERIMENTS.append({
    "id": "A9_All7_mean", "label": "Todos los Fractales (mean)", "group": "A",
    "type": "sm", "subset": "All_Fractals",
})

# --- Group B: Per-Channel, All-16 (6 subsets, like Martinez-Peon Table 2) ---
pc_configs = [
    ("B1_HRS_pc", "HRS p64 (per-channel)", ["HRS_p64"]),
    ("B2_HO_pc", "HO (per-channel)", ["HO"]),
    ("B3_HV_pc", "HV (per-channel)", ["HV"]),
    ("B4_Martinez_pc", "Martinez HO+HRS+HV (per-channel)", PER_CH_METHODS_MARTINEZ),
    ("B5_Basic_pc", "Basicos (per-channel)", PER_CH_METHODS_BASIC),
    ("B6_All7_pc", "Todos 7 (per-channel)", PER_CH_METHODS_ALL),
]
for sid, label, methods in pc_configs:
    EXPERIMENTS.append({
        "id": sid, "label": label, "group": "B", "type": "pc",
        "methods": methods,
    })

# --- Group C: Per-Channel with Channel Subsets ---
pc_subset_configs = [
    ("Martinez", PER_CH_METHODS_MARTINEZ),
    ("Basicos", PER_CH_METHODS_BASIC),
    ("Todos7", PER_CH_METHODS_ALL),
]
ch_subset_keys = [k for k in CHANNEL_SUBSETS_CONFIG if k != "All-16"]
for label_base, methods in pc_subset_configs:
    for ch_key in ch_subset_keys:
        chs = CHANNEL_SUBSETS_CONFIG[ch_key]
        nch = len(chs)
        nm = len(methods)
        sid = f"C_{label_base}_{ch_key}"
        lbl = f"{label_base} per-ch ({ch_key}, {nch}ch x {nm}m)"
        EXPERIMENTS.append({
            "id": sid, "label": lbl, "group": "C", "type": "pc_subset",
            "methods": methods, "channels": [ch_key],
        })

MONTHS = [("Mixto", None), ("Mes 1", "Month1"), ("Mes 3", "Month3"), ("Mes 6", "Month6")]

# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("CLASIFICACION FRACTAL EXTENDIDA - Martinez-Peon 2024 Style + SMOTE")
    print("=" * 80)
    n_total = len(EXPERIMENTS) * len(MODELS) * len(MONTHS) * 2
    print(f"Experimentos: {len(EXPERIMENTS)} subsets x 7 modelos x 4 condiciones x 2 tipos = {n_total}")
    print(f"  Group A (spatial mean): {sum(1 for e in EXPERIMENTS if e['group']=='A')} subsets")
    print(f"  Group B (per-channel):  {sum(1 for e in EXPERIMENTS if e['group']=='B')} subsets")
    print(f"  Group C (ch subsets):   {sum(1 for e in EXPERIMENTS if e['group']=='C')} subsets")
    print(f"  SMOTE: 3-class only (train folds)")
    print("=" * 80)

    csv_path = os.path.join(cfg.FEATURES_DIR, "Fixed_W700_3class_Features.csv")
    df_raw = pd.read_csv(csv_path)
    print(f"\nLoaded: {len(df_raw)} epochs, {len(df_raw.columns)} columns")
    print(f"Classes: {dict(sorted(df_raw['MVR_Class'].value_counts().items()))}")
    print(f"Sessions per month: {dict(df_raw['Session'].value_counts().sort_index())}")
    print()

    t0 = time.time()
    results = []
    done = 0

    for month_label, month_val in MONTHS:
        for exp in EXPERIMENTS:
            for ct in ["3class", "2class"]:
                done += 1

                # Build features
                try:
                    if exp["type"] in ("sm", "sm_custom"):
                        X, y, feat_cols, df_filt = _extract_xy_spatial_mean(df_raw, exp, month_val)
                    elif exp["type"] in ("pc", "pc_subset"):
                        X, y, feat_cols, df_filt = _extract_xy_per_channel(df_raw, exp, month_val)
                    else:
                        continue
                except Exception as e:
                    print(f"[{done}/{n_total}] SKIP {exp['id']:25s}: {e}")
                    continue

                if len(feat_cols) < 1 or len(y) < 10:
                    continue

                # Handle class type
                if ct == "2class":
                    mask_2c = np.isin(y, (10, 40))
                    X_c, y_c = X[mask_2c], y[mask_2c]
                    pat = df_filt.iloc[mask_2c]["Patient"].values
                else:
                    X_c, y_c = X.copy(), y.copy()
                    pat = df_filt["Patient"].values

                if len(y_c) < 10 or len(np.unique(y_c)) < 2:
                    continue

                use_smote = (ct == "3class")

                for mname in MODEL_ORDER:
                    model = MODELS[mname]
                    all_yt, all_yp = _classify_patient_cv(X_c, y_c, pat, model, use_smote)

                    if len(all_yt) > 0:
                        acc = accuracy_score(all_yt, all_yp)
                        f1 = f1_score(all_yt, all_yp, average="macro", zero_division=0)
                        kappa = cohen_kappa_score(all_yt, all_yp)
                        mae = mean_absolute_error(all_yt, all_yp)
                        results.append({
                            "Model": mname, "Feature_Subset": exp["id"],
                            "Feature_Label": exp["label"], "Channel_Config": exp.get("channels", ["All-16"])[0],
                            "Class_Type": ct, "Month": month_label,
                            "Accuracy": acc, "F1_Macro": f1, "Kappa": kappa, "MAE": mae,
                            "N": len(all_yt), "N_Features": len(feat_cols), "Group": exp["group"],
                        })

                # Progress
                last_batch = [r for r in results if r["Feature_Subset"] == exp["id"]
                              and r["Month"] == month_label and r["Class_Type"] == ct]
                best_acc = max(r["Accuracy"] for r in last_batch) if last_batch else 0
                elapsed = time.time() - t0
                eta = (elapsed / done) * (n_total - done) if done > 0 else 0
                pct = done / n_total * 100
                print(f"[{done:4d}/{n_total}] {exp['id']:25s} {ct:7s} {month_label:6s} | "
                      f"{pct:5.1f}% | ETA {eta/60:5.1f}m | Best={best_acc:.4f}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min. {len(results)} results saved.")

    df_r = pd.DataFrame(results)
    out_path = os.path.join(cfg.CLASSIFICATION_DIR, "results_W700_fractal_extended.csv")
    df_r.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # ===== PRINT MARTINEZ-PEON STYLE TABLES =====
    sub_3c = df_r[df_r["Class_Type"] == "3class"].copy()
    sm_ids = [e["id"] for e in EXPERIMENTS if e["group"] == "A" and "A9" not in e["id"]]
    sm_labels = {e["id"]: e["label"] for e in EXPERIMENTS if e["group"] == "A"}
    pc_ids = [e["id"] for e in EXPERIMENTS if e["group"] == "B"]
    pc_labels = {e["id"]: e["label"] for e in EXPERIMENTS if e["group"] == "B"}

    SHORT_PC = ["HRS p64/ch", "HO/ch", "HV/ch", "HRS+HO+HV/ch", "Basic/ch", "All7/ch"]
    SHORT_SM = ["RS", "Higuchi", "DFA", "Semivar.", "Promed.", "HO", "HRS_p64", "HV"]

    print("\n" + "=" * 120)
    print("TABLA 5.1: SPATIAL MEAN - ALGORITMOS INDIVIDUALES (3-clases, SMOTE)")
    print("=" * 120)
    sm_all_ids = [e["id"] for e in EXPERIMENTS if e["group"] == "A"]
    for month in ["Mixto", "Mes 1", "Mes 3", "Mes 6"]:
        ms = sub_3c[(sub_3c["Month"] == month) & (sub_3c["Feature_Subset"].isin(sm_all_ids))]
        if len(ms) == 0:
            continue
        print(f"\n--- {month} ---")
        header = f"{'Clasificador':<22s}"
        for sid in sm_all_ids:
            lbl = sm_labels[sid]
            header += f" {lbl[:7]:>8s}"
        print(header)
        for mname in MODEL_ORDER:
            line = f"{mname:<22s}"
            for sid in sm_all_ids:
                row = ms[(ms["Model"] == mname) & (ms["Feature_Subset"] == sid)]
                if len(row) > 0:
                    line += f" {row.iloc[0]['Accuracy']*100:>7.2f}%"
                else:
                    line += f" {'-':>8s}"
            print(line)

    print("\n" + "=" * 120)
    print("TABLA 5.2: PER-CHANNEL ALL-16 (3-clases, SMOTE) - ESTILO MARTINEZ-PEON TABLE 2")
    print("=" * 120)
    for month in ["Mixto", "Mes 1", "Mes 3", "Mes 6"]:
        ms = sub_3c[(sub_3c["Month"] == month) & (sub_3c["Feature_Subset"].isin(pc_ids))]
        if len(ms) == 0:
            continue
        print(f"\n--- {month} ---")
        header = f"{'Clasificador':<22s}"
        for sn in SHORT_PC:
            header += f" {sn:>14s}"
        print(header)
        header2 = f"{'':22s}"
        for _ in SHORT_PC:
            header2 += f" {'Acc':>6s} {'kappa':>6s}"
        print(header2)
        for mname in MODEL_ORDER:
            line = f"{mname:<22s}"
            for sid in pc_ids:
                row = ms[(ms["Model"] == mname) & (ms["Feature_Subset"] == sid)]
                if len(row) > 0:
                    v = row.iloc[0]
                    line += f"  {v['Accuracy']*100:5.2f}% {v['Kappa']:+.3f}"
                else:
                    line += f" {'-':>14s}"
            print(line)

    print("\n" + "=" * 120)
    print("MEJORES RESULTADOS POR CONDICION (3-clases, SMOTE)")
    print("=" * 120)
    for month in ["Mixto", "Mes 1", "Mes 3", "Mes 6"]:
        ms = sub_3c[sub_3c["Month"] == month]
        if len(ms) == 0:
            continue
        best = ms.loc[ms["Accuracy"].idxmax()]
        print(f"{month:6s}: {best['Model']:<18s} + {best['Feature_Label']:<50s} "
              f"= {best['Accuracy']*100:.2f}%  F1={best['F1_Macro']:.4f}  "
              f"k={best['Kappa']:.4f}  MAE={best['MAE']:.4f}  N_feat={best['N_Features']}")

    print("\n" + "=" * 120)
    print("MEJORES 2-CLASES (sin SMOTE)")
    print("=" * 120)
    sub_2c = df_r[df_r["Class_Type"] == "2class"]
    for month in ["Mixto", "Mes 1", "Mes 3", "Mes 6"]:
        ms = sub_2c[sub_2c["Month"] == month]
        if len(ms) == 0:
            continue
        best = ms.loc[ms["Accuracy"].idxmax()]
        print(f"{month:6s}: {best['Model']:<18s} + {best['Feature_Label']:<50s} "
              f"= {best['Accuracy']*100:.2f}%  F1={best['F1_Macro']:.4f}  "
              f"\u03ba={best['Kappa']:.4f}  N_feat={best['N_Features']}")

    print("\n" + "=" * 120)
    print("CHANNEL SUBSETS - MEJOR POR CONFIG (Group C, 3-clases, SMOTE)")
    print("=" * 120)
    group_c = sub_3c[sub_3c["Group"] == "C"]
    for month in ["Mixto", "Mes 1", "Mes 3", "Mes 6"]:
        ms = group_c[group_c["Month"] == month]
        if len(ms) == 0:
            continue
        best_c = ms.loc[ms["Accuracy"].idxmax()]
        print(f"\n{month}: {best_c['Model']} + {best_c['Feature_Label']} = {best_c['Accuracy']*100:.2f}%")
        # Top 5 per channel config
        top = ms.nlargest(5, "Accuracy")
        for _, row in top.iterrows():
            print(f"  {row['Model']:<18s} {row['Feature_Label']:<55s} {row['Accuracy']*100:>6.2f}%")

    print("\nSaved:", out_path)
