"""Per-patient (intra-subject) analysis of the W700 pipeline, used for the congress poster.

Reproduces the intra-subject CV of classify_w700_full.py (Delta+Spectral, All-16,
StratifiedKFold k=5 per patient, StandardScaler fit on train only) with
Logistic Regression, but keeps the predictions of each patient separately so the
month-to-month evolution can be reported per patient.

For every (balancing mode x month x patient) it reports:
- Accuracy and BALANCED accuracy (mean per-class recall). Balanced accuracy is the
  reported metric: plain accuracy is inflated when a recording has few rest epochs,
  because rest is the class the model recognizes worst (see INFORME_INTRASUJETO.md).
- Fold SD of balanced accuracy (error bars of the poster figure).
- Macro F1 and per-class recall (TPR rest / 10% / 40%).
- Per-patient confusion matrices (JSON), used for the averaged confusion matrix.

Balancing modes (applied ONLY to the training folds, never before splitting):
- none:    no balancing (poster results)
- smote:   SMOTE oversampling of the training fold (imbalanced-learn)
- weights: LogisticRegression(class_weight="balanced")

Input:  output/features/Fixed_W700_3class_Features.csv (from extract_w700_full.py)
Output: output/classification/results_W700_intrasubject_per_patient.csv
        output/classification/cms_W700_intrasubject_LR_3class.json
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from src.classification.features import get_channel_subset_features, get_feature_subset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score)
from imblearn.over_sampling import SMOTE

PATIENT_MAP = {"Px.006": "P1", "Px.007": "P2", "Px.008": "P3", "Px.009": "P4", "Px.010": "P5"}
MONTHS = ["Month1", "Month3", "Month6"]
FEATURE_SUBSET = "Delta+Spectral"
CHANNEL_CONF = "All-16"
MODES = ["none", "smote", "weights"]
CLASS_TYPES = {"3class": [0, 10, 40], "2class": [10, 40]}


def _patient_cv(X, y, labels, mode):
    """Intra-subject StratifiedKFold for one patient. Same n_splits formula as
    src.classification.validation.run_intra_subject_pool_cv."""
    n_splits = min(5, max(2, len(y) // 3))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_bal, yt_all, yp_all = [], [], []
    for tr, te in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_te = scaler.transform(X[te])
        y_tr = y[tr]
        if mode == "smote":
            X_tr, y_tr = SMOTE(random_state=42).fit_resample(X_tr, y_tr)
        model = LogisticRegression(max_iter=2000, random_state=42,
                                   class_weight="balanced" if mode == "weights" else None)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        fold_bal.append(balanced_accuracy_score(y[te], y_pred))
        yt_all.extend(y[te])
        yp_all.extend(y_pred)
    return np.array(yt_all), np.array(yp_all), fold_bal


def _row(mode, class_type, month, patient, yt, yp, labels, fold_bal=None):
    cm = confusion_matrix(yt, yp, labels=labels)
    rec = np.diag(cm) / cm.sum(axis=1)
    row = {
        "Mode": mode, "Class_Type": class_type, "Month": month, "Patient": patient,
        "N": len(yt),
        "Accuracy": accuracy_score(yt, yp) * 100,
        "Balanced_Accuracy": balanced_accuracy_score(yt, yp) * 100,
        "BalAcc_Fold_SD": np.std(fold_bal, ddof=1) * 100 if fold_bal else np.nan,
        "F1_Macro": f1_score(yt, yp, average="macro", zero_division=0),
    }
    names = {0: "Rest", 10: "MVR10", 40: "MVR40"}
    for lab, r in zip(labels, rec):
        row["TPR_" + names[lab]] = r * 100
    for lab in labels:
        row["N_" + names[lab]] = int((yt == lab).sum())
    return row, cm


def main():
    csv_path = os.path.join(cfg.FEATURES_DIR, "Fixed_W700_3class_Features.csv")
    df = get_channel_subset_features(pd.read_csv(csv_path), CHANNEL_CONF)

    rows, cms = [], {}
    for mode in MODES:
        for class_type, labels in CLASS_TYPES.items():
            if class_type == "2class" and mode != "none":
                continue  # 2-class is balanced (~1:1); no resampling needed
            for month in MONTHS:
                d = df[df["Session"] == month]
                X, y, _, df_f = get_feature_subset(d, FEATURE_SUBSET)
                pat = df_f["Patient"].values
                mask = np.isin(y, labels)
                X, y, pat = X[mask], y[mask], pat[mask]

                pool_t, pool_p = [], []
                for p in sorted(set(pat)):
                    m = pat == p
                    yt, yp, fb = _patient_cv(X[m], y[m], labels, mode)
                    row, cm = _row(mode, class_type, month, PATIENT_MAP[p], yt, yp, labels, fb)
                    rows.append(row)
                    if class_type == "3class":
                        cms[f"{mode}|{month}|{PATIENT_MAP[p]}"] = cm.tolist()
                    pool_t.extend(yt)
                    pool_p.extend(yp)
                row, _ = _row(mode, class_type, month, "Group",
                              np.array(pool_t), np.array(pool_p), labels)
                rows.append(row)

    out = pd.DataFrame(rows)
    out_csv = os.path.join(cfg.CLASSIFICATION_DIR, "results_W700_intrasubject_per_patient.csv")
    out.to_csv(out_csv, index=False)
    out_json = os.path.join(cfg.CLASSIFICATION_DIR, "cms_W700_intrasubject_LR_3class.json")
    with open(out_json, "w") as fh:
        json.dump(cms, fh, indent=1)

    pd.set_option("display.width", 200)
    show = out[(out.Class_Type == "3class")]
    print(show.pivot_table(index=["Month", "Patient"], columns="Mode",
                           values="Balanced_Accuracy").round(2).to_string())
    print("Saved:", out_csv)
    print("Saved:", out_json)


if __name__ == "__main__":
    main()
