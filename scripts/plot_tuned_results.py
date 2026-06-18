"""Regenerate key figures with SVM tuned results.
Focus: evolution_months.png showing SVM convergence Mes 1->3->6.
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Publication style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 14, "axes.labelsize": 13,
    "figure.dpi": 100, "savefig.dpi": 200,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})

MONTH_LABELS = ["Month 1", "Month 3", "Month 6"]
MONTH_POSITIONS = [1, 3, 6]

# SVM results per month (3-class, Basic per-channel 64 feat, SMOTE)
# Mes 1: default SVM from classify_fractal_extended (no tuning done)
# Mes 3,6: tuned from hyperparameter_tuning
SVM_ACC = {
    "Month 1": 69.90,
    "Month 3": 84.27,
    "Month 6": 85.46,
}

SVM_STD = {
    "Month 1": 9.1,
    "Month 3": 4.9,
    "Month 6": 5.1,
}

# LogReg (previous best, for comparison)
LOGREG_ACC = {
    "Month 1": 74.80,
    "Month 3": 83.27,
    "Month 6": 83.99,
}
LOGREG_STD = {
    "Mes 1": 11.4,
    "Mes 3": 4.5,
    "Mes 6": 5.4,
}

# SVM default (before tuning, for comparison)
SVM_DEFAULT_ACC = {
    "Month 1": 69.90,
    "Month 3": 79.36,
    "Month 6": 80.38,
}

# Per-patient tuned SVM (if available) or use defaults
PATIENT_LABELS = ["P1", "P2", "P3", "P4", "P5"]


def plot_svm_evolution():
    fig, ax = plt.subplots(figsize=(8, 5))

    months_idx = [1, 3, 6]
    svm_acc = [SVM_ACC[m] for m in MONTH_LABELS]
    svm_std = [SVM_STD[m] for m in MONTH_LABELS]
    logreg_acc = [LOGREG_ACC[m] for m in MONTH_LABELS]

    svm_lower = [a - s for a, s in zip(svm_acc, svm_std)]
    svm_upper = [a + s for a, s in zip(svm_acc, svm_std)]

    ax.fill_between(months_idx, svm_lower, svm_upper, alpha=0.12,
                    color="#2166AC")
    ax.plot(months_idx, svm_acc,
            marker="o", markersize=11, linewidth=2.5,
            color="#2166AC", label="SVM (best)", zorder=4)
    ax.plot(months_idx, logreg_acc,
            marker="s", markersize=9, linewidth=2, linestyle="--",
            color="#B2182B", label="Logistic Regression (2nd)", zorder=3)

    for i, (m, v) in enumerate(zip(months_idx, svm_acc)):
        ax.annotate(f"{v:.1f}%", (m, v + 3.5), ha="center", fontsize=11,
                    fontweight="bold", color="#333333")

    ax.set_xticks(months_idx)
    ax.set_xticklabels(MONTH_LABELS, fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_ylim(60, 94)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_title("SVM Classification Accuracy Over Post-Stroke Recovery\n(Fractal per-channel features + SMOTE, 3-class)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_xlim(0.5, 6.5)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "evolution_months.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_spatial_vs_perchannel_updated():
    """Spatial mean vs per-channel comparison with tuned values."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    for idx, month in enumerate(["Month 1", "Month 3", "Month 6"]):
        ax = axes[idx]
        categories = ["Spatial Mean\n(best, 7 feat)", "Per-channel\n(default, 64 feat)", "Per-channel\n(SVM, 64 feat)"]

        if month == "Month 1":
            values = [51.12, 69.90, 69.90]
        elif month == "Month 3":
            values = [64.64, 79.36, 84.27]
        else:
            values = [57.61, 80.38, 85.46]

        colors = ["#D1E5F0", "#92C5DE", "#2166AC"]
        bars = ax.bar(categories, values, color=colors, edgecolor="white", linewidth=1.2, width=0.6)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")

        ax.set_title(month, fontsize=13, fontweight="bold")
        ax.set_ylim(0, 95)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.grid(True, alpha=0.3, axis="y", linestyle=":")
        ax.tick_params(axis="x", labelsize=9)

    axes[0].set_ylabel("Accuracy (%)", fontsize=13)
    fig.suptitle("Spatial Mean vs Per-channel Features\n(Basic fractal features + SMOTE, 3-class)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "spatial_vs_perchannel.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_tuning_impact():
    """Bar chart: default vs tuned for SVM across months."""
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(3)
    width = 0.3

    defaults = [SVM_DEFAULT_ACC[m] for m in MONTH_LABELS]
    tuned = [SVM_ACC[m] for m in MONTH_LABELS]

    bars1 = ax.bar(x - width / 2, defaults, width, label="SVM (default)", color="#D1E5F0", edgecolor="white")
    bars2 = ax.bar(x + width / 2, tuned, width, label="SVM (optimized)", color="#2166AC", edgecolor="white")

    for bar, v in zip(bars1, defaults):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", fontsize=9)
    for bar, v in zip(bars2, tuned):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold", color="#2166AC")

    # Deltas
    for i, (d, t) in enumerate(zip(defaults, tuned)):
        delta = t - d
        ax.annotate(f"+{delta:.1f} pts", (x[i] + width / 2, t + 3),
                    ha="center", fontsize=10, fontweight="bold", color="#2166AC",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(MONTH_LABELS, fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_title("SVM: default vs. optimized hyperparameters\n(Basic per-channel features + SMOTE, 3-class)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y", linestyle=":")
    ax.set_ylim(60, 94)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "tuning_impact.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    print("Regenerating key figures with SVM tuned results...")
    plot_svm_evolution()
    plot_spatial_vs_perchannel_updated()
    plot_tuning_impact()
    print("Done.")
