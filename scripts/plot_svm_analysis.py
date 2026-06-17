"""Generate confusion matrix and per-patient analysis with tuned SVM."""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

PATIENT_MAP = {"Px.006": "P1", "Px.007": "P2", "Px.008": "P3", "Px.009": "P4", "Px.010": "P5"}

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 14, "axes.labelsize": 13,
    "figure.dpi": 100, "savefig.dpi": 200,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})

OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, "plots")
ALL_CHANNELS = cfg.CHANNEL_NAMES
PER_CH_METHODS_BASIC = ["RS", "Higuchi", "DFA", "Variogram"]
CLASS_LABELS = ["Rest (0%)", "MVR 10%", "MVR 40%"]


def build_features(df_raw):
    cols = []
    for ch in ALL_CHANNELS:
        for m in PER_CH_METHODS_BASIC:
            col = f"Active_{ch}_{m}"
            if col in df_raw.columns:
                cols.append(col)
    return cols


def evaluate_svm_per_patient(X, y, pat, C=12, gamma=0.02):
    """SVM tuned, SMOTE, per-patient CV. Returns pool yt, yp, per-patient accs."""
    model = SVC(C=C, kernel="rbf", gamma=gamma, random_state=42, probability=True,
                cache_size=2000)
    all_yt, all_yp = [], []
    patient_accs = {}
    sm = SMOTE(random_state=42)

    for patient in sorted(set(pat)):
        p_mask = pat == patient
        X_p, y_p = X[p_mask], y[p_mask]
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
            patient_accs[patient] = accuracy_score(patient_yt, patient_yp)
    return all_yt, all_yp, patient_accs


def plot_confusion_matrix_tuned(cm, accuracy):
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=100)
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Percentage (%)", fontsize=11)

    for i in range(3):
        for j in range(3):
            text = f"{cm[i, j]}\n({cm_norm[i, j]:.1f}%)"
            color = "white" if cm_norm[i, j] > 50 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=12,
                    fontweight="bold", color=color)

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_LABELS, fontsize=11)
    ax.set_yticklabels(CLASS_LABELS, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=13, fontweight="bold")
    ax.set_ylabel("Actual", fontsize=13, fontweight="bold")
    ax.set_title(f"Confusion Matrix — SVM tuned (C=12, rbf, gamma=0.02)\n3-class, Mes 6, Pool Accuracy = {accuracy:.2f}%", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_matrix_best.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_accuracy_per_patient_tuned(patient_accs, month_label):
    patients_raw = list(patient_accs.keys())
    patients = [PATIENT_MAP.get(p, p) for p in patients_raw]
    accs = [patient_accs[p] * 100 for p in patients_raw]
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#D1E5F0", "#92C5DE", "#4393C3", "#2166AC", "#053061"]
    bars = ax.bar(range(len(patients)), accs, color=colors[:len(patients)], edgecolor="white", linewidth=1.5)
    ax.axhline(y=mean_acc, color="#B2182B", linestyle="--", linewidth=2, alpha=0.7,
               label=f"Mean: {mean_acc:.1f}%")
    ax.fill_between([-0.5, len(patients) - 0.5], mean_acc - std_acc, mean_acc + std_acc,
                    alpha=0.1, color="#B2182B")
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{v:.1f}%", ha="center", fontsize=12, fontweight="bold", color="#2166AC")
    ax.set_xticks(range(len(patients)))
    ax.set_xticklabels(patients, fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title(f"Per-Patient Accuracy — SVM tuned\n{month_label}, 3-class, Mean = {mean_acc:.1f}% ± {std_acc:.1f}%", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.set_ylim(60, 100)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.grid(True, alpha=0.3, axis="y", linestyle=":")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "accuracy_per_patient.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    csv_path = os.path.join(cfg.FEATURES_DIR, "Fixed_W700_3class_Features.csv")
    df_raw = pd.read_csv(csv_path)
    feat_cols = build_features(df_raw)

    MONTHS = [("Mes 3", "Month3"), ("Mes 6", "Month6"),
              ("Mes 1", "Month1"), ("Mixto", None)]

    for month_label, month_val in MONTHS:
        df_f = df_raw.copy()
        if month_val:
            df_f = df_f[df_f["Session"] == month_val].copy()
        y = df_f["MVR_Class"].values
        X = df_f[feat_cols].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        pat = df_f["Patient"].values

        # Use best params per month
        if month_label == "Mes 6":
            C, gamma = 12, 0.02
        elif month_label == "Mes 3":
            C, gamma = 100, 0.01
        else:
            C, gamma = 10, "scale"

        all_yt, all_yp, patient_accs = evaluate_svm_per_patient(
            X, y, pat, C=C, gamma=gamma)

        if len(all_yt) == 0:
            continue

        pool_acc = accuracy_score(all_yt, all_yp) * 100
        cm = confusion_matrix(all_yt, all_yp)
        mean_pat = np.mean(list(patient_accs.values())) * 100
        std_pat = np.std(list(patient_accs.values())) * 100

        print(f"{month_label}: Pool={pool_acc:.2f}%  PatMean={mean_pat:.2f}%+-{std_pat:.1f}  "
              f"Per-patient: {{{', '.join(f'{PATIENT_MAP.get(k,k)}: {v*100:.1f}%' for k, v in patient_accs.items())}}}")

        if month_label == "Mes 6":
            plot_confusion_matrix_tuned(cm, pool_acc)
            plot_accuracy_per_patient_tuned(patient_accs, month_label)


if __name__ == "__main__":
    main()
