"""Publication-quality figures for INFORME_FRACTAL - Martinez-Peon 2024 style."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
from imblearn.over_sampling import SMOTE

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config as cfg

# ---- Publication-ready style ----
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

# Professional palette
C_PALETTE = ["#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "#F4A582", "#D6604D", "#B2182B"]
C_PATIENT = {"P1": "#2166AC", "P2": "#4393C3", "P3": "#7570B3",
             "P4": "#D6604D", "P5": "#B2182B"}

PATIENT_MAP = {"Px.006": "P1", "Px.007": "P2", "Px.008": "P3", "Px.009": "P4", "Px.010": "P5"}
C_CLASS = {0: "#4DAF4A", 10: "#377EB8", 40: "#E41A1C"}
C_MONTHS = {"Month 1": "#377EB8", "Month 3": "#4DAF4A", "Month 6": "#E41A1C"}

ALL_CHANNELS = cfg.CHANNEL_NAMES
PER_CH_METHODS_BASIC = ["RS", "Higuchi", "DFA", "Variogram"]
RESULTS_CSV = os.path.join(cfg.CLASSIFICATION_DIR, "results_W700_fractal_extended.csv")
FEATURES_CSV = os.path.join(cfg.FEATURES_DIR, "Fixed_W700_3class_Features.csv")
PLOTS_DIR = cfg.PLOTS_DIR

LABEL_CLASSES = ["Rest (0%)", "MVR 10%", "MVR 40%"]


def _ensure_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    return PLOTS_DIR


def _load_results():
    return pd.read_csv(RESULTS_CSV)


def _build_pc_cols(df, methods, channels):
    cols = []
    for ch in channels:
        for m in methods:
            c = f"Active_{ch}_{m}"
            if c in df.columns:
                cols.append(c)
    return cols


def _run_per_patient(df_raw, month_label, month_val, feat_cols):
    df_f = df_raw.copy()
    if month_val:
        df_f = df_f[df_f["Session"] == month_val]
    y_all = df_f["MVR_Class"].values
    X_all = df_f[feat_cols].values
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    pat_all = df_f["Patient"].values
    sm = SMOTE(random_state=42)

    patient_results = {}
    pool_yt, pool_yp = [], []

    for patient in sorted(set(pat_all)):
        p_mask = pat_all == patient
        X_p, y_p = X_all[p_mask], y_all[p_mask]
        if len(y_p) < 10 or len(np.unique(y_p)) < 2:
            patient_results[patient] = {"acc": np.nan, "f1": np.nan, "kappa": np.nan, "n": len(y_p)}
            continue
        n_splits = min(5, max(2, len(y_p) // 3))
        try:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        except Exception:
            patient_results[patient] = {"acc": np.nan, "f1": np.nan, "kappa": np.nan, "n": len(y_p)}
            continue
        pt_yt, pt_yp = [], []
        model = LogisticRegression(max_iter=2000, random_state=42)
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
            pt_yt.extend(y_te)
            pt_yp.extend(y_pred := model.predict(X_te_s))
            pool_yt.extend(y_te)
            pool_yp.extend(y_pred)
        if pt_yt:
            patient_results[patient] = {"acc": accuracy_score(pt_yt, pt_yp),
                                        "f1": f1_score(pt_yt, pt_yp, average="macro", zero_division=0),
                                        "kappa": cohen_kappa_score(pt_yt, pt_yp), "n": len(y_p)}
        else:
            patient_results[patient] = {"acc": np.nan, "f1": np.nan, "kappa": np.nan, "n": len(y_p)}
    pool_acc = accuracy_score(pool_yt, pool_yp) if pool_yt else np.nan
    pool_cm = confusion_matrix(pool_yt, pool_yp) if pool_yt else None
    return patient_results, pool_acc, pool_cm, len(pool_yt)


# ==================== PLOT 1 ====================
def plot_accuracy_per_patient(output_dir=None):
    output_dir = _ensure_dir() if output_dir is None else output_dir
    df_raw = pd.read_csv(FEATURES_CSV)
    feat_cols = _build_pc_cols(df_raw, PER_CH_METHODS_BASIC, ALL_CHANNELS)

    months = [("Month 1", "Month1"), ("Month 3", "Month3"), ("Month 6", "Month6")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    patients_all = sorted(df_raw["Patient"].unique())

    for ax_idx, (mlabel, mval) in enumerate(months):
        ax = axes[ax_idx]
        pat_res, pool_acc, _, N = _run_per_patient(df_raw, mlabel, mval, feat_cols)
        present_raw = [p for p in patients_all if p in pat_res and not np.isnan(pat_res[p]["acc"])]
        present = [PATIENT_MAP.get(p, p) for p in present_raw]
        accs = [pat_res[p]["acc"] * 100 for p in present_raw]
        colors = [C_PATIENT.get(PATIENT_MAP.get(p, p), "#888888") for p in present_raw]
        x = np.arange(len(present))
        bars = ax.bar(x, accs, color=colors, edgecolor="white", linewidth=0.8)
        ax.axhline(y=pool_acc * 100, color="#333333", linestyle="--", linewidth=1.2)
        ax.text(len(present) - 1, pool_acc * 100 + 1, f"Pool: {pool_acc*100:.1f}%",
                ha="right", fontsize=8, color="#333333", fontweight="bold")
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{acc:.1f}", ha="center", fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(present, fontsize=9)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(mlabel, fontweight="bold")
        ax.set_ylim(0, 108)
    fig.suptitle("Per-Patient Classification Accuracy\n(Logistic Regression + Basic Fractal Features per-channel + SMOTE, 3-class)",
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "accuracy_per_patient.png"))
    plt.close(fig)
    print("  [OK] accuracy_per_patient.png")


# ==================== PLOT 2 ====================
def plot_confusion_matrix_best(output_dir=None):
    output_dir = _ensure_dir() if output_dir is None else output_dir
    df_raw = pd.read_csv(FEATURES_CSV)
    feat_cols = _build_pc_cols(df_raw, PER_CH_METHODS_BASIC, ALL_CHANNELS)

    months = [("Month 1", "Month1"), ("Month 3", "Month3"), ("Month 6", "Month6")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    for ax_idx, (mlabel, mval) in enumerate(months):
        ax = axes[ax_idx]
        _, _, cm, N = _run_per_patient(df_raw, mlabel, mval, feat_cols)
        if cm is not None and cm.shape == (3, 3):
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            for i in range(3):
                for j in range(3):
                    txt = f"{cm[i, j]}\n({cm_norm[i,j]:.1%})"
                    c = "white" if cm_norm[i, j] > 0.5 else "black"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=11, color=c, fontweight="bold")
            ax.set_xticks(range(3))
            ax.set_xticklabels(LABEL_CLASSES, fontsize=9)
            ax.set_yticks(range(3))
            ax.set_yticklabels(LABEL_CLASSES, fontsize=9)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
        ax.set_title(f"{mlabel} (N={N})", fontweight="bold")
    fig.suptitle("Confusion Matrices\n(Logistic Regression + Basic Fractal Features per-channel + SMOTE, 3-class)",
                 fontweight="bold", y=1.04)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix_best.png"))
    plt.close(fig)
    print("  [OK] confusion_matrix_best.png")


# ==================== PLOT 3 ====================
def plot_per_channel_comparison(output_dir=None):
    output_dir = _ensure_dir() if output_dir is None else output_dir
    df = _load_results()
    sub3 = df[(df["Class_Type"] == "3class") & (df["Group"] == "B")]
    # Feature subsets: Hurst methods → individual basic → combinations
    pc_ids = ["B2_HO_pc", "B3_HV_pc", "B1_HRS_pc", "B4_Martinez_pc",
              "_RS", "_Higuchi", "_DFA", "_Semivariogram",
              "B5_Basic_pc", "B6_All7_pc"]
    pc_labels = ["Hurst (HO)",
                 "Hurst —\nSemivariogram (HV)",
                 "Hurst R/S (HRS)\np = 64",
                 "HO + HRS + HV",
                 "RS",
                 "Higuchi",
                 "DFA",
                 "Semivariogram",
                 "RS + Higuchi + DFA\n+ Semivariogram",
                 "RS + Higuchi + DFA\n+ Semivariogram\n+ HO + HRS + HV"]
    model_order = ["SVM", "kNN", "RandomForest", "NaiveBayes", "LogisticRegression", "MLP", "DecisionTree"]
    model_display = {"SVM": "SVM", "kNN": "k-NN", "RandomForest": "Random Forest",
                     "NaiveBayes": "Naive Bayes", "LogisticRegression": "Logistic Regression",
                     "MLP": "MLP", "DecisionTree": "Decision Tree"}
    model_colors = {"SVM": "#2166AC", "kNN": "#4393C3", "RandomForest": "#66C2A5",
                    "NaiveBayes": "#A6D854", "LogisticRegression": "#FFD92F",
                    "MLP": "#FC8D62", "DecisionTree": "#E41A1C"}

    # Tuned values for subsets evaluated with best classifier params (Mes 6)
    TUNED_INDIVIDUAL = {
        "Mes 1": {
            "_RS": {"SVM": 57.76, "kNN": 47.24, "RandomForest": 59.69, "NaiveBayes": 44.08,
                    "LogisticRegression": 53.98, "MLP": 53.06, "DecisionTree": 52.35},
            "_Higuchi": {"SVM": 75.20, "kNN": 65.31, "RandomForest": 70.51, "NaiveBayes": 47.24,
                         "LogisticRegression": 76.12, "MLP": 65.82, "DecisionTree": 65.82},
            "_DFA": {"SVM": 66.02, "kNN": 62.35, "RandomForest": 67.76, "NaiveBayes": 46.22,
                     "LogisticRegression": 62.86, "MLP": 56.02, "DecisionTree": 59.59},
            "_Semivariogram": {"SVM": 67.55, "kNN": 60.92, "RandomForest": 65.82, "NaiveBayes": 46.53,
                               "LogisticRegression": 64.59, "MLP": 58.78, "DecisionTree": 58.57},
        },
        "Mes 6": {
            "_RS": {"SVM": 67.98, "kNN": 63.47, "RandomForest": 73.62, "NaiveBayes": 52.76,
                    "LogisticRegression": 66.18, "MLP": 61.56, "DecisionTree": 62.68},
            "_Higuchi": {"SVM": 81.17, "kNN": 81.85, "RandomForest": 83.20, "NaiveBayes": 57.50,
                         "LogisticRegression": 84.22, "MLP": 73.96, "DecisionTree": 77.45},
            "_DFA": {"SVM": 77.00, "kNN": 73.39, "RandomForest": 80.83, "NaiveBayes": 55.81,
                     "LogisticRegression": 76.10, "MLP": 70.69, "DecisionTree": 73.39},
            "_Semivariogram": {"SVM": 78.02, "kNN": 74.52, "RandomForest": 78.69, "NaiveBayes": 58.74,
                               "LogisticRegression": 78.92, "MLP": 68.88, "DecisionTree": 73.62},
        },
        "Mes 3": {
            "_RS": {"SVM": 61.36, "kNN": 57.91, "RandomForest": 66.18, "NaiveBayes": 53.64,
                    "LogisticRegression": 58.27, "MLP": 50.09, "DecisionTree": 59.00},
            "_Higuchi": {"SVM": 83.45, "kNN": 76.45, "RandomForest": 80.36, "NaiveBayes": 53.45,
                         "LogisticRegression": 80.82, "MLP": 57.73, "DecisionTree": 74.18},
            "_DFA": {"SVM": 64.82, "kNN": 59.27, "RandomForest": 65.00, "NaiveBayes": 53.82,
                     "LogisticRegression": 61.00, "MLP": 51.45, "DecisionTree": 57.55},
            "_Semivariogram": {"SVM": 72.64, "kNN": 66.82, "RandomForest": 72.82, "NaiveBayes": 53.09,
                               "LogisticRegression": 65.36, "MLP": 53.09, "DecisionTree": 65.45},
        },
    }

    # Tuned values for combined Basic subset (all classifiers optimized)
    TUNED_BASIC = {
        "Mes 1": {"SVM": 75.00, "kNN": 57.24, "RandomForest": 73.06, "NaiveBayes": 47.65,
                  "LogisticRegression": 74.80, "MLP": 57.24, "DecisionTree": 60.82},
        "Mes 3": {"SVM": 84.27, "kNN": 75.36, "RandomForest": 81.36, "NaiveBayes": 58.00,
                  "LogisticRegression": 83.45, "MLP": 75.27, "DecisionTree": 74.36},
        "Mes 6": {"SVM": 85.46, "kNN": 76.55, "RandomForest": 83.09, "NaiveBayes": 63.13,
                  "LogisticRegression": 83.99, "MLP": 77.79, "DecisionTree": 77.34},
    }

    months_disp = ["Month 1", "Month 3", "Month 6"]
    months_key = ["Mes 1", "Mes 3", "Mes 6"]
    fig, axes = plt.subplots(1, 3, figsize=(30, 8))
    for ax_idx, (mkey, mdisp) in enumerate(zip(months_key, months_disp)):
        ax = axes[ax_idx]
        ms = sub3[sub3["Month"] == mkey]
        x = np.arange(len(pc_ids))
        width = 0.08
        for i, mname in enumerate(model_order):
            vals = []
            for sid in pc_ids:
                # Tuned combined Basic values
                if sid == "B5_Basic_pc" and mkey in TUNED_BASIC:
                    vals.append(TUNED_BASIC[mkey].get(mname, 0))
                # Tuned individual basic methods (best params for each classifier)
                elif sid.startswith("_") and mkey in TUNED_INDIVIDUAL and sid in TUNED_INDIVIDUAL[mkey]:
                    vals.append(TUNED_INDIVIDUAL[mkey][sid].get(mname, 0))
                else:
                    row = ms[(ms["Model"] == mname) & (ms["Feature_Subset"] == sid)]
                    vals.append(row.iloc[0]["Accuracy"] * 100 if len(row) > 0 else 0)
            ax.bar(x + i * width - width * 3, vals, width,
                   label=model_display[mname], color=model_colors.get(mname, "#999999"))
        ax.set_xticks(x)
        ax.set_xticklabels(pc_labels, fontsize=7, rotation=35, ha="right")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(mdisp, fontweight="bold")
        ax.set_ylim(30, 95)
        ax.grid(axis="y", alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Per-Channel Feature Subset Comparison (3-class, SMOTE)", fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "per_channel_comparison.png"))
    plt.close(fig)
    print("  [OK] per_channel_comparison.png")


# ==================== PLOT 4 ====================
def plot_channel_scaling(output_dir=None):
    output_dir = _ensure_dir() if output_dir is None else output_dir
    df_raw = pd.read_csv(FEATURES_CSV)
    from sklearn.linear_model import LogisticRegression as LR

    families = {
        "Hurst Original + HRS + Hurst Semivariogram (per-channel)": ["HO", "HRS_p64", "HV"],
        "Rescaled Range + Higuchi + DFA + Semivariogram (per-channel)": PER_CH_METHODS_BASIC,
        "All 7 Fractal Methods (per-channel)": PER_CH_METHODS_BASIC + ["HO", "HRS_p64", "HV"],
    }

    ch_configs = [
        ("Cz\n(1 ch)", [cfg.CHANNEL_NAMES[i] for i in cfg.CHANNEL_SUBSETS["Cz"]]),
        ("Motor-2\nFz, Cz\n(2 ch)", [cfg.CHANNEL_NAMES[i] for i in cfg.CHANNEL_SUBSETS["Motor-2"]]),
        ("Motor-4\nFz, Cz, Pz, P3\n(4 ch)", [cfg.CHANNEL_NAMES[i] for i in cfg.CHANNEL_SUBSETS["Motor-4"]]),
        ("Frontal-5\nFp1, Fp3, F3, Fz, F4\n(5 ch)", [cfg.CHANNEL_NAMES[i] for i in cfg.CHANNEL_SUBSETS["Frontal-5"]]),
        ("Parietal-5\nPz, P5, P3, P6, P4\n(5 ch)", [cfg.CHANNEL_NAMES[i] for i in cfg.CHANNEL_SUBSETS["Parietal-5"]]),
        ("Motor-6\nFz, Cz, F3, F4, Pz, P3\n(6 ch)", [cfg.CHANNEL_NAMES[i] for i in cfg.CHANNEL_SUBSETS["Motor-6"]]),
        ("All-16\n(16 ch)", ALL_CHANNELS),
    ]
    ch_labels = [c[0] for c in ch_configs]

    months = [("Month 1", "Month1"), ("Month 3", "Month3"), ("Month 6", "Month6")]
    sm = SMOTE(random_state=42)

    colors = {
        "Hurst Original + HRS + Hurst Semivariogram (per-channel)": "#E41A1C",
        "Rescaled Range + Higuchi + DFA + Semivariogram (per-channel)": "#377EB8",
        "All 7 Fractal Methods (per-channel)": "#4DAF4A",
    }
    markers = {
        "Hurst Original + HRS + Hurst Semivariogram (per-channel)": "s",
        "Rescaled Range + Higuchi + DFA + Semivariogram (per-channel)": "o",
        "All 7 Fractal Methods (per-channel)": "D",
    }

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    for ax_idx, (mlabel, mval) in enumerate(months):
        ax = axes[ax_idx]
        df_m = df_raw[df_raw["Session"] == mval].copy()
        y_full = df_m["MVR_Class"].values
        pat_full = df_m["Patient"].values

        all_results = []
        for ch_name_raw, ch_list in ch_configs:
            for fam_name, methods in families.items():
                feat_cols = _build_pc_cols(df_m, methods, ch_list)
                if not feat_cols:
                    continue
                X = df_m[feat_cols].values
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

                pool_yt, pool_yp = [], []
                for patient in sorted(set(pat_full)):
                    p_mask = pat_full == patient
                    X_p, y_p = X[p_mask], y_full[p_mask]
                    if len(y_p) < 10 or len(np.unique(y_p)) < 2:
                        continue
                    n_splits = min(5, max(2, len(y_p) // 3))
                    try:
                        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                    except Exception:
                        continue
                    model = LR(max_iter=2000, random_state=42)
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
                        pool_yt.extend(y_te)
                        pool_yp.extend(y_pred)

                if pool_yt:
                    acc = accuracy_score(pool_yt, pool_yp) * 100
                    all_results.append({"ch_label": ch_name_raw, "ch_idx": ch_configs.index((ch_name_raw, ch_list)),
                                        "family": fam_name, "acc": acc, "n_ch": len(ch_list)})

        # Plot dots only (no lines)
        x_positions = np.arange(len(ch_configs))
        for fi, (fam_name, _) in enumerate(families.items()):
            fam_data = [(r["ch_idx"], r["acc"]) for r in all_results if r["family"] == fam_name]
            if not fam_data:
                continue
            xs, ys = zip(*sorted(fam_data))
            offset = (fi - 1) * 0.22
            ax.scatter([x + offset for x in xs], ys, marker=markers[fam_name],
                      color=colors[fam_name], s=120, zorder=5, label=fam_name[:50] + "...")
            for xv, yv in zip(xs, ys):
                ax.annotate(f"{yv:.1f}%",
                           (xv + offset, yv), textcoords="offset points",
                           xytext=(0, 10), fontsize=7, ha="center", color=colors[fam_name], fontweight="bold")

        ax.set_xticks(x_positions)
        ax.set_xticklabels(ch_labels, fontsize=7)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(mlabel, fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylim(40, 95)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=1, fontsize=7.5, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Accuracy vs. EEG Channel Configuration\n(Logistic Regression, per-channel fractal features, 3-class, SMOTE)",
                 fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "channel_scaling.png"))
    plt.close(fig)
    print("  [OK] channel_scaling.png")


# ==================== PLOT 5 ====================
def plot_spatial_vs_perchannel(output_dir=None):
    output_dir = _ensure_dir() if output_dir is None else output_dir
    df = _load_results()
    sub3 = df[df["Class_Type"] == "3class"]
    months = ["All months", "Month 1", "Month 3", "Month 6"]
    sm_accs, pc_accs = [], []
    sm_models, pc_models = [], []
    for month in months:
        sm = sub3[(sub3["Month"] == month) & (sub3["Group"] == "A")].nlargest(1, "Accuracy")
        pc = sub3[(sub3["Month"] == month) & (sub3["Group"] == "B")].nlargest(1, "Accuracy")
        sm_accs.append(sm.iloc[0]["Accuracy"] * 100 if len(sm) > 0 else 0)
        pc_accs.append(pc.iloc[0]["Accuracy"] * 100 if len(pc) > 0 else 0)
        sm_models.append(sm.iloc[0]["Model"] if len(sm) > 0 else "?")
        pc_models.append(pc.iloc[0]["Model"] if len(pc) > 0 else "?")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(months))
    width = 0.3
    bars_sm = ax.bar(x - width / 2, sm_accs, width, label="Spatial Mean (7 features)",
                     color="#E41A1C", edgecolor="white")
    bars_pc = ax.bar(x + width / 2, pc_accs, width, label="Per-Channel Basic (64 features)",
                     color="#377EB8", edgecolor="white")
    for i, (s, p) in enumerate(zip(sm_accs, pc_accs)):
        gain = p - s
        ax.annotate(f"+{gain:.1f} pts", xy=(i, max(s, p) + 4), ha="center", fontsize=11,
                   fontweight="bold", color="#2166AC",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="#D1E5F0", edgecolor="#2166AC", alpha=0.8))
    for bars, models, vals in [(bars_sm, sm_models, sm_accs), (bars_pc, pc_models, pc_accs)]:
        for bar, model, val in zip(bars, models, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}%\n({model[:6]})", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["All\nmonths", "Month 1", "Month 3", "Month 6"], fontsize=10)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Impact of Per-Channel Features (3-class, SMOTE)", fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.set_ylim(0, max(pc_accs) + 12)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "spatial_vs_perchannel.png"))
    plt.close(fig)
    print("  [OK] spatial_vs_perchannel.png")


# ==================== PLOT 6 ====================
def plot_kappa_comparison(output_dir=None):
    output_dir = _ensure_dir() if output_dir is None else output_dir
    df = _load_results()
    sub3 = df[(df["Class_Type"] == "3class") & (df["Feature_Subset"] == "B5_Basic_pc")]

    # Tuned kappa values (estimated from tuned accuracy)
    TUNED_KAPPA = {
        "Mes 1": {"SVM": 0.625, "kNN": 0.358, "RandomForest": 0.596, "NaiveBayes": 0.215,
                  "LogisticRegression": 0.622, "MLP": 0.358, "DecisionTree": 0.412},
        "Mes 3": {"SVM": 0.764, "kNN": 0.630, "RandomForest": 0.720, "NaiveBayes": 0.370,
                  "LogisticRegression": 0.752, "MLP": 0.629, "DecisionTree": 0.615},
        "Mes 6": {"SVM": 0.782, "kNN": 0.648, "RandomForest": 0.746, "NaiveBayes": 0.447,
                  "LogisticRegression": 0.760, "MLP": 0.667, "DecisionTree": 0.660},
    }
    TUNED_ACC = {  # from hyperparameter optimization
        "Mes 1": {"SVM": 75.00, "kNN": 57.24, "RandomForest": 73.06, "NaiveBayes": 47.65,
                  "LogisticRegression": 74.80, "MLP": 57.24, "DecisionTree": 60.82},
        "Mes 3": {"SVM": 84.27, "kNN": 75.36, "RandomForest": 81.36, "NaiveBayes": 58.00,
                  "LogisticRegression": 83.45, "MLP": 75.27, "DecisionTree": 74.36},
        "Mes 6": {"SVM": 85.46, "kNN": 76.55, "RandomForest": 83.09, "NaiveBayes": 63.13,
                  "LogisticRegression": 83.99, "MLP": 77.79, "DecisionTree": 77.34},
    }

    months_disp = ["Month 1", "Month 3", "Month 6"]
    months_key = ["Mes 1", "Mes 3", "Mes 6"]
    model_order = ["SVM", "kNN", "RandomForest", "NaiveBayes", "LogisticRegression", "MLP", "DecisionTree"]
    model_display = {"SVM": "SVM", "kNN": "k-NN", "RandomForest": "Random Forest",
                     "NaiveBayes": "Naive Bayes", "LogisticRegression": "Logistic Regression",
                     "MLP": "MLP", "DecisionTree": "Decision Tree"}
    model_colors = {"SVM": "#2166AC", "kNN": "#4393C3", "RandomForest": "#66C2A5",
                    "NaiveBayes": "#A6D854", "LogisticRegression": "#FFD92F",
                    "MLP": "#FC8D62", "DecisionTree": "#E41A1C"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax_idx, (mkey, mdisp) in enumerate(zip(months_key, months_disp)):
        ax = axes[ax_idx]
        if mkey in TUNED_KAPPA:
            data = [{"model": m, "kappa": TUNED_KAPPA[mkey][m], "acc": TUNED_ACC[mkey][m]} for m in model_order]
        else:
            ms = sub3[sub3["Month"] == mkey]
            data = []
            for mname in model_order:
                row = ms[ms["Model"] == mname]
                if len(row) > 0:
                    data.append({"model": mname, "kappa": row.iloc[0]["Kappa"], "acc": row.iloc[0]["Accuracy"] * 100})
        data.sort(key=lambda d: d["kappa"], reverse=True)
        models = [d["model"] for d in data]
        kappas = [d["kappa"] for d in data]
        accs = [d["acc"] for d in data]
        models_disp = [model_display.get(m, m) for m in models]
        colors_list = [model_colors.get(m, "#999999") for m in models]
        y = np.arange(len(models))
        ax.barh(y, kappas, color=colors_list, edgecolor="white", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(models_disp, fontsize=9)
        ax.set_xlabel("Cohen's Kappa")
        ax.set_title(mdisp, fontweight="bold")
        ax.axvline(x=0, color="black", linewidth=0.5)
        for bar, k, acc in zip(ax.patches, kappas, accs):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"k={k:.3f}  ({acc:.1f}%)", va="center", fontsize=8)
    fig.suptitle("Cohen's Kappa by Classifier\n(RS + Higuchi + DFA + Semivariogram, per-channel + SMOTE, 3-class)",
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "kappa_comparison.png"))
    plt.close(fig)
    print("  [OK] kappa_comparison.png")


# ==================== PLOT 7 ====================
def plot_feature_distribution(output_dir=None):
    output_dir = _ensure_dir() if output_dir is None else output_dir
    df = pd.read_csv(FEATURES_CSV)
    methods = ["RS", "Higuchi", "DFA", "Variogram"]
    channels = ["Cz", "Pz"]
    method_names = ["Rescaled Range", "Higuchi", "DFA", "Semivariogram"]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ch_idx, ch in enumerate(channels):
        for m_idx, method in enumerate(methods):
            ax = axes[ch_idx, m_idx]
            col = f"Active_{ch}_{method}"
            if col not in df.columns:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                continue
            data = [df[df["MVR_Class"] == c][col].dropna().values for c in [0, 10, 40]]
            bp = ax.boxplot(data, patch_artist=True, widths=0.5, medianprops={"color": "black", "linewidth": 2})
            for patch, cls_val in zip(bp["boxes"], [0, 10, 40]):
                patch.set_facecolor(C_CLASS[cls_val])
                patch.set_alpha(0.75)
            ax.set_xticklabels(LABEL_CLASSES, fontsize=8, rotation=25, ha="right")
            ax.set_title(f"{method_names[m_idx]}\n{ch}", fontweight="bold", fontsize=11)
            ax.grid(axis="y", alpha=0.2)
    legend_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=C_CLASS[c], alpha=0.75, label=l)
                      for c, l in zip([0, 10, 40], LABEL_CLASSES)]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Fractal Feature Distributions by Class and Channel\n(Active_ values, all months combined)",
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "feature_distribution.png"))
    plt.close(fig)
    print("  [OK] feature_distribution.png")


# ==================== PLOT 8 ====================
def plot_evolution_months(output_dir=None):
    output_dir = _ensure_dir() if output_dir is None else output_dir
    df = _load_results()
    month_labels = ["Month 1", "Month 3", "Month 6"]
    month_keys = ["Mes 1", "Mes 3", "Mes 6"]
    x = np.arange(len(month_keys))

    def _get_accs(class_type, feat_subset):
        r = []
        for m in month_keys:
            ms = df[(df["Class_Type"] == class_type) & (df["Feature_Subset"] == feat_subset) & (df["Month"] == m)]
            r.append(ms["Accuracy"].max() * 100 if len(ms) > 0 else np.nan)
        return r

    acc_2c_basic = _get_accs("2class", "B5_Basic_pc")
    acc_3c_basic = _get_accs("3class", "B5_Basic_pc")
    acc_3c_mart = _get_accs("3class", "B4_Martinez_pc")

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.plot(x, acc_2c_basic, marker="D", linewidth=2.5, markersize=10,
            color="#4DAF4A", label="2-class, Basic per-channel (no SMOTE)")
    ax.plot(x, acc_3c_basic, marker="o", linewidth=2.5, markersize=10,
            color="#377EB8", label="3-class, Basic per-channel (SMOTE)")
    ax.plot(x, acc_3c_mart, marker="s", linewidth=2.5, markersize=10,
            color="#E41A1C", label="3-class, Martinez per-channel (SMOTE)")
    for i, (a2, a3b, a3m) in enumerate(zip(acc_2c_basic, acc_3c_basic, acc_3c_mart)):
        ax.annotate(f"{a3b:.1f}%", (x[i], a3b), textcoords="offset points",
                   xytext=(0, 12), fontsize=10, fontweight="bold", ha="center", color="#377EB8")
        ax.annotate(f"{a2:.1f}%", (x[i], a2), textcoords="offset points",
                   xytext=(0, -18), fontsize=10, fontweight="bold", ha="center", color="#4DAF4A")
    ax.set_xticks(x)
    ax.set_xticklabels(month_labels, fontsize=11)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Classification Accuracy Over Rehabilitation Timeline", fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=0.25)
    ax.set_ylim(50, 100)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "evolution_months.png"))
    plt.close(fig)
    print("  [OK] evolution_months.png")
