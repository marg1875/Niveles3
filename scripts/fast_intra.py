"""Fast intra-subject classification test."""
import os, sys, warnings, time
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, r"C:\Users\Desktop-UHC57\OneDrive\Escritorio\Niveles3")
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

csv_path = os.path.join(cfg.FEATURES_DIR, "Global_All_Features.csv")
df_raw = pd.read_csv(csv_path)

channel_configs = ["Cz", "Motor-2", "Motor-4", "Motor-6", "All-16"]
feature_subsets = ["Delta", "Delta+Martinez", "Martinez"]
models = {
    "SVM": SVC(random_state=42, probability=True),
    "RandomForest": RandomForestClassifier(random_state=42, n_jobs=-1),
    "LogReg": LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1),
    "kNN": KNeighborsClassifier(n_neighbors=10, metric="chebyshev", n_jobs=-1),
    "NaiveBayes": GaussianNB(),
}

results = []
for ch_conf in channel_configs:
    df_ch = get_channel_subset_features(df_raw, ch_conf)
    for feat_subset in feature_subsets:
        try:
            X, y, feat_cols, df_filt = get_feature_subset(df_ch, feat_subset)
        except:
            continue
        if len(feat_cols) < 2:
            continue
        for class_type in ["3class", "2class"]:
            if class_type == "2class":
                mask = np.isin(y, (10, 40))
                X_c, y_c = X[mask], y[mask]
                pat = df_filt.iloc[mask]["Patient"].values
            else:
                X_c, y_c = X, y
                pat = df_filt["Patient"].values

            for mname, m in models.items():
                all_y_true, all_y_pred = [], []
                for patient in sorted(set(pat)):
                    p_idx = (pat == patient)
                    X_p, y_p = X_c[p_idx], y_c[p_idx]
                    if len(y_p) < 10 or len(set(y_p)) < 2:
                        continue
                    n_splits = min(5, max(2, len(y_p) // 3))
                    try:
                        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                    except:
                        continue
                    for tr, te in skf.split(X_p, y_p):
                        X_tr, X_te = X_p[tr], X_p[te]
                        y_tr, y_te = y_p[tr], y_p[te]
                        scaler = StandardScaler()
                        X_tr_s = scaler.fit_transform(X_tr)
                        X_te_s = scaler.transform(X_te)
                        m.fit(X_tr_s, y_tr)
                        y_pred = m.predict(X_te_s)
                        all_y_true.extend(y_te)
                        all_y_pred.extend(y_pred)

                if len(all_y_true) > 0:
                    acc = accuracy_score(all_y_true, all_y_pred)
                    f1 = f1_score(all_y_true, all_y_pred, average="macro", zero_division=0)
                    results.append({
                        "Model": mname, "Channel_Config": ch_conf,
                        "Feature_Subset": feat_subset, "Class_Type": class_type,
                        "Accuracy": acc, "F1": f1, "N": len(all_y_true)
                    })
                    print(f"  {mname:20s} {ch_conf:10s} {feat_subset:16s} {class_type:6s} Acc={acc:.4f} F1={f1:.4f}")

df_r = pd.DataFrame(results)
df_r.to_csv(os.path.join(cfg.CLASSIFICATION_DIR, "results_intra_fast.csv"), index=False)
print(f"\nResults saved. Best intra-subject 3-class:")
best3 = df_r[df_r.Class_Type == "3class"].nlargest(5, "Accuracy")
print(best3[["Model", "Channel_Config", "Feature_Subset", "Accuracy", "F1"]].to_string(index=False))
print(f"\nBest intra-subject 2-class:")
best2 = df_r[df_r.Class_Type == "2class"].nlargest(5, "Accuracy")
print(best2[["Model", "Channel_Config", "Feature_Subset", "Accuracy", "F1"]].to_string(index=False))
