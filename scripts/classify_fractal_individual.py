"""Classification with individual fractal algorithms (1 feature each, Martinez-Peon style)."""
import os, sys, time, warnings, numpy as np, pandas as pd
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
from sklearn.metrics import accuracy_score, f1_score

MODELS = {
    "SVM": SVC(random_state=42, probability=True),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1),
    "kNN": KNeighborsClassifier(n_neighbors=10, metric="chebyshev", n_jobs=-1),
    "NaiveBayes": GaussianNB(),
}

SUBSETS = [
    "RS_only", "Higuchi_only", "DFA_only", "Semivariograma_only",
    "HO_only", "HRS_only", "HV_only", "All_Fractals",
]
CHANNEL_CONF = "All-16"
MONTHS = [("Mixto", None), ("Mes 1", "Month1"), ("Mes 3", "Month3"), ("Mes 6", "Month6")]

csv_path = os.path.join(cfg.FEATURES_DIR, "Fixed_W700_3class_Features.csv")
df_raw = pd.read_csv(csv_path)
print("Loaded {} epochs (classes: {})".format(len(df_raw), dict(df_raw["MVR_Class"].value_counts())))

results = []
total = len(MODELS) * len(SUBSETS) * len(MONTHS) * 2  # 3class + 2class
done = 0

for month_label, month_val in MONTHS:
    df_ch = get_channel_subset_features(df_raw, CHANNEL_CONF)
    if month_val:
        df_ch = df_ch[df_ch["Session"] == month_val].copy()

    for fs_name in SUBSETS:
        try:
            X, y, feat_cols, df_filt = get_feature_subset(df_ch, fs_name)
        except Exception:
            continue
        if len(feat_cols) < 1:
            continue

        for ct in ["3class", "2class"]:
            if ct == "2class":
                mask = np.isin(y, (10, 40))
                X_c, y_c = X[mask], y[mask]
                pat = df_filt.iloc[mask]["Patient"].values
            else:
                X_c, y_c = X, y
                pat = df_filt["Patient"].values

            if len(y_c) < 10 or len(np.unique(y_c)) < 2:
                continue

            for mname, model in MODELS.items():
                done += 1
                all_yt, all_yp = [], []
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
                        scaler = StandardScaler()
                        X_tr_s = scaler.fit_transform(X_tr)
                        X_te_s = scaler.transform(X_te)
                        model.fit(X_tr_s, y_tr)
                        y_pred = model.predict(X_te_s)
                        all_yt.extend(y_te)
                        all_yp.extend(y_pred)

                if len(all_yt) > 0:
                    acc = accuracy_score(all_yt, all_yp)
                    f1 = f1_score(all_yt, all_yp, average="macro", zero_division=0)
                    results.append({
                        "Model": mname, "Feature_Subset": fs_name,
                        "Class_Type": ct, "Month": month_label,
                        "Accuracy": acc, "F1": f1, "N": len(all_yt),
                    })
                    print("[{}/{}] {:20s} {:22s} {:8s} {:6s} Acc={:.4f} F1={:.4f} N={}".format(
                        done, total, mname, fs_name, month_label, ct, acc, f1, len(all_yt)))

df_r = pd.DataFrame(results)
out_path = os.path.join(cfg.CLASSIFICATION_DIR, "results_W700_fractal_individual.csv")
df_r.to_csv(out_path, index=False)

# Print Martinez-Peon style tables (3-class)
print("\n" + "=" * 100)
print("TABLAS ESTILO MARTINEZ-PEON (3-clases: 0%, 10%, 40%)")
print("=" * 100)

sub_3c = df_r[df_r["Class_Type"] == "3class"]
alg_names = ["Rescaled Range", "Higuchi", "DFA", "Semivariograma", "Hurst Original",
             "Hurst R/S Particiones", "Hurst Semivariograma", "Todos los Fractales"]
fs_map = dict(zip(SUBSETS, alg_names))
model_order = ["SVM", "RandomForest", "LogisticRegression", "kNN", "NaiveBayes"]

for month in ["Mixto", "Mes 1", "Mes 3", "Mes 6"]:
    ms = sub_3c[sub_3c["Month"] == month]
    print("\n--- {} ---".format(month))
    header = "{:<20s}".format("Clasificador")
    for fs in SUBSETS:
        header += " {:>10s}".format(fs_map[fs])
    print(header)
    for model in model_order:
        line = "{:<20s}".format(model)
        for fs in SUBSETS:
            row = ms[(ms["Model"] == model) & (ms["Feature_Subset"] == fs)]
            if len(row) > 0:
                line += " {:>9.2f}%".format(row.iloc[0]["Accuracy"] * 100)
            else:
                line += " {:>10s}".format("-")
        print(line)

# Best per month
print("\n" + "=" * 100)
print("MEJOR RESULTADO POR MES (3-clases)")
for month in ["Mixto", "Mes 1", "Mes 3", "Mes 6"]:
    ms = sub_3c[sub_3c["Month"] == month]
    best = ms.loc[ms["Accuracy"].idxmax()]
    print("{}: {} + {} = {:.2f}% F1={:.4f} N={}".format(
        month, best["Model"], fs_map[best["Feature_Subset"]],
        best["Accuracy"] * 100, best["F1"], best["N"]))

# 2-class comparison
print("\n" + "=" * 100)
print("COMPARACION 2-CLASES (sin reposo)")
sub_2c = df_r[df_r["Class_Type"] == "2class"]
for month in ["Mixto", "Mes 1", "Mes 3", "Mes 6"]:
    ms = sub_2c[sub_2c["Month"] == month]
    best = ms.loc[ms["Accuracy"].idxmax()]
    print("{}: {} + {} = {:.2f}%".format(month, best["Model"], fs_map[best["Feature_Subset"]], best["Accuracy"] * 100))

print("\nSaved: " + out_path)
