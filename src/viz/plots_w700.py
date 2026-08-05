"""
Publication figures for INFORME_COMPLETO.md (the W700 fixed-window
pipeline: scripts/extract_w700_full.py + scripts/classify_w700_full.py).

This is a separate module from src/viz/plots_fractal.py on purpose —
plots_fractal.py covers INFORME_FRACTAL.md's SMOTE-based 3-class,
per-channel-only track (a different results file, different methodology).
Nothing in that module is imported, edited, or reused here, so existing
hand-tuned figures stay untouched.

Reads output/classification/results_W700_full_intra.csv (written by
classify_w700_full.py) directly — no live re-classification needed except
for the confusion matrix, whose per-prediction data isn't in that CSV
(only aggregate Accuracy/F1 are).
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import config as cfg

# Same "publication-ready" convention as src/viz/plots_fractal.py (kept as a
# separate copy rather than a shared import — see module docstring).
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

C_2CLASS = "#2166AC"
C_3CLASS = "#D6604D"
C_FRACTAL = "#92C5DE"
C_SPECTRAL = "#4DAF4A"
C_COMBINED = "#2166AC"

RESULTS_CSV = os.path.join(cfg.CLASSIFICATION_DIR, "results_W700_full_intra.csv")
FEATURES_CSV = os.path.join(cfg.FEATURES_DIR, "Fixed_W700_3class_Features.csv")

MONTH_ORDER = ["Mixed", "Month1", "Month3", "Month6"]
MONTH_DISP = {"Mixed": "Mixto", "Month1": "Mes 1\n(agudo)", "Month3": "Mes 3\n(sub-agudo)",
              "Month6": "Mes 6\n(cronico)"}


def _ensure_dir(output_dir):
    output_dir = output_dir or cfg.PLOTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _load_results():
    return pd.read_csv(RESULTS_CSV)


def plot_accuracy_by_month(output_dir=None):
    """Best accuracy per month, Delta+Spectral subset, 2-class vs 3-class —
    the headline result table (INFORME_COMPLETO.md 7.1 / 5.7) as a figure."""
    output_dir = _ensure_dir(output_dir)
    df = _load_results()
    sub = df[df["Feature_Subset"] == "Delta+Spectral"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(MONTH_ORDER))
    width = 0.35

    acc_2c, acc_3c = [], []
    for month in MONTH_ORDER:
        s2 = sub[(sub["Month"] == month) & (sub["Class_Type"] == "2class")]
        s3 = sub[(sub["Month"] == month) & (sub["Class_Type"] == "3class")]
        acc_2c.append(s2["Accuracy"].max() * 100 if len(s2) else np.nan)
        acc_3c.append(s3["Accuracy"].max() * 100 if len(s3) else np.nan)

    bars1 = ax.bar(x - width / 2, acc_2c, width, label="2-clases (10% vs 40%)", color=C_2CLASS)
    bars2 = ax.bar(x + width / 2, acc_3c, width, label="3-clases (Reposo/10%/40%)", color=C_3CLASS)
    for bar, v in zip(bars1, acc_2c):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.1f}%",
                ha="center", fontsize=9, fontweight="bold", color=C_2CLASS)
    for bar, v in zip(bars2, acc_3c):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.1f}%",
                ha="center", fontsize=9, fontweight="bold", color=C_3CLASS)

    ax.set_xticks(x)
    ax.set_xticklabels([MONTH_DISP[m] for m in MONTH_ORDER])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_title("Accuracy por mes — Δ+Espectral (18 caracteristicas)\nW700, validacion intra-sujeto 5-fold CV",
                 fontweight="bold")
    # Bars run from y=0 up to 60-95%, so there's no clear space left inside
    # the axes for a legend without overlapping some bar's value label —
    # placed below the x-axis instead.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, framealpha=1.0)
    ax.grid(True, axis="y", alpha=0.3, linestyle=":")
    fig.subplots_adjust(bottom=0.22)

    path = os.path.join(output_dir, "w700_accuracy_by_month.png")
    fig.savefig(path)
    plt.close(fig)
    print("  [OK] w700_accuracy_by_month.png")


def plot_fractal_vs_spectral(output_dir=None):
    """Fractal-only vs Spectral-only vs Combined, per month — one panel for
    2-class, one for 3-class (INFORME_COMPLETO.md §5.6 / §5.7 as a figure)."""
    output_dir = _ensure_dir(output_dir)
    df = _load_results()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    titles = {"2class": "2-clases (10% vs 40%)", "3class": "3-clases (Reposo/10%/40%)"}

    for ax, class_type in zip(axes, ["2class", "3class"]):
        sub = df[df["Class_Type"] == class_type]
        x = np.arange(len(MONTH_ORDER))
        width = 0.25

        frac, spec, comb = [], [], []
        for month in MONTH_ORDER:
            s = sub[sub["Month"] == month]
            frac.append(s[s["Feature_Subset"] == "Delta+Martinez"]["Accuracy"].max() * 100)
            spec.append(s[s["Feature_Subset"] == "Spectral"]["Accuracy"].max() * 100)
            comb.append(s[s["Feature_Subset"] == "Delta+Spectral"]["Accuracy"].max() * 100)

        ax.bar(x - width, frac, width, label="Solo Fractales (8)", color=C_FRACTAL)
        ax.bar(x, spec, width, label="Solo Espectral (13)", color=C_SPECTRAL)
        ax.bar(x + width, comb, width, label="Δ+Espectral (18)", color=C_COMBINED)

        ax.set_xticks(x)
        ax.set_xticklabels([MONTH_DISP[m] for m in MONTH_ORDER], fontsize=8)
        ax.set_title(titles[class_type], fontweight="bold")
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.grid(True, axis="y", alpha=0.3, linestyle=":")

    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.suptitle("Contribucion de Fractales vs Espectrales — W700", fontweight="bold", y=1.02)
    fig.tight_layout()

    path = os.path.join(output_dir, "w700_fractal_vs_spectral.png")
    fig.savefig(path)
    plt.close(fig)
    print("  [OK] w700_fractal_vs_spectral.png")


def plot_rest_epoch_distribution(output_dir=None):
    """Genuine rest-epoch counts by month (Apendice B.2 as a figure) — shows
    the class-imbalance context behind the 3-class results."""
    output_dir = _ensure_dir(output_dir)
    df = pd.read_csv(FEATURES_CSV)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(MONTH_ORDER))
    width = 0.6

    rest, active_10, active_40 = [], [], []
    for month in MONTH_ORDER:
        sub = df if month == "Mixed" else df[df["Session"] == month]
        counts = sub["MVR_Class"].value_counts()
        rest.append(counts.get(0, 0))
        active_10.append(counts.get(10, 0))
        active_40.append(counts.get(40, 0))

    b1 = ax.bar(x, rest, width, label="Reposo (0%)", color="#4DAF4A")
    b2 = ax.bar(x, active_10, width, bottom=rest, label="MVR 10%", color="#377EB8")
    bottom_10 = np.array(rest) + np.array(active_10)
    b3 = ax.bar(x, active_40, width, bottom=bottom_10, label="MVR 40%", color="#E41A1C")

    totals = np.array(rest) + np.array(active_10) + np.array(active_40)
    for xi, (r, t) in enumerate(zip(rest, totals)):
        pct = 100 * r / t if t else 0
        ax.text(xi, t + 40, f"{t}\n({pct:.0f}% reposo)", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([MONTH_DISP[m] for m in MONTH_ORDER])
    ax.set_ylabel("Numero de epocas")
    ax.set_ylim(0, max(totals) * 1.18)  # headroom for the total/% labels above each bar
    ax.set_title("Distribucion de clases por mes (W700)\nReposo detectado por huecos inter-evento > 5s",
                 fontweight="bold")
    # "Mixto" is always the tallest stack (sum of all months), so an
    # in-axes legend anywhere near the top collides with its label —
    # placed below the x-axis instead (same fix as plot_accuracy_by_month).
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, framealpha=1.0)
    ax.grid(True, axis="y", alpha=0.3, linestyle=":")
    fig.subplots_adjust(bottom=0.2)

    path = os.path.join(output_dir, "w700_class_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    print("  [OK] w700_class_distribution.png")


def plot_confusion_matrix_3class_best(output_dir=None):
    """Confusion matrix for the best 3-class result (Month 6, LogReg,
    Spectral — 67.55% acc / F1=0.5711, see INFORME_COMPLETO.md §5.7).

    results_W700_full_intra.csv only stores aggregate Accuracy/F1, not
    per-prediction labels, so this retrains that one specific experiment via
    the same shared CV helper classify_w700_full.py uses
    (run_intra_subject_pool_cv) to get pooled (y_true, y_pred) for the matrix.
    Same model/params/random_state as the reported result — reproduces it
    exactly, not an approximation.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
    from src.classification.features import get_channel_subset_features, get_feature_subset
    from src.classification.validation import run_intra_subject_pool_cv

    output_dir = _ensure_dir(output_dir)
    df_raw = pd.read_csv(FEATURES_CSV)
    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns
    df_raw[numeric_cols] = np.nan_to_num(df_raw[numeric_cols].values, nan=0.0, posinf=0.0, neginf=0.0)

    df_ch = get_channel_subset_features(df_raw, "All-16")
    df_ch = df_ch[df_ch["Session"] == "Month6"].copy()
    X, y, feat_cols, df_filt = get_feature_subset(df_ch, "Spectral")
    pat = df_filt["Patient"].values

    model = LogisticRegression(max_iter=2000, random_state=42, n_jobs=1)
    y_true, y_pred = run_intra_subject_pool_cv(X, y, pat, model)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    labels = [0, 10, 40]
    label_disp = ["Reposo\n(0%)", "MVR\n10%", "MVR\n40%"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(label_disp)
    ax.set_yticklabels(label_disp)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    for i in range(3):
        for j in range(3):
            color = "white" if cm_norm[i, j] > 55 else "black"
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.1f}%)", ha="center", va="center",
                     fontsize=10, color=color, fontweight="bold" if i == j else "normal")
    ax.set_title(f"Matriz de Confusion — LogisticRegression\n3-clases, Mes 6, Espectral, "
                 f"Accuracy={acc*100:.2f}% F1={f1:.4f}", fontweight="bold", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="% de la clase real")

    path = os.path.join(output_dir, "w700_confusion_matrix_3class_month6.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] w700_confusion_matrix_3class_month6.png (Acc={acc*100:.2f}%, F1={f1:.4f})")
