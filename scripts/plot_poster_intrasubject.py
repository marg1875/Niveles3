"""Poster figures for the intra-subject analysis (style of Martinez-Peon et al. 2024, Fig. 8).

(a) Balanced accuracy per patient (x = P1..P5), one line per month with
    +/- SD across the 5 CV folds, chance level 33 %.
(b) Confusion matrix of month 6, row-normalized per patient and averaged
    across patients (mean +/- SD). The mean of its diagonal equals the
    balanced accuracy of panel (a) averaged over patients.

Reads the outputs of scripts/analyze_intrasubject_w700.py (mode "none").
Writes output/plots/poster_fig8a_balanced_accuracy.png and
       output/plots/poster_fig8b_confusion_month6.png
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import to_rgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
                     "font.size": 26})

MODE = "none"
PATIENTS = ["P1", "P2", "P3", "P4", "P5"]
MONTHS = ["Month1", "Month3", "Month6"]
MONTH_LABELS = ["Mes 1", "Mes 3", "Mes 6"]
MONTH_COLORS = ["#8FB3E8", "#3F74C9", "#102F6E"]   # light -> dark = time order
MONTH_MARKERS = ["o", "s", "D"]


def plot_accuracy_per_patient(res, out_path):
    r = res[(res.Mode == MODE) & (res.Class_Type == "3class")]
    fig, ax = plt.subplots(figsize=(16.19, 7.6), dpi=200)
    ax.axhline(100 / 3, color="#C00000", lw=2, ls=(0, (6, 5)), alpha=.6, zorder=1)
    ax.text(4.4, 100 / 3 + 1.0, "nivel de azar (33 %)", ha="right", va="bottom",
            fontsize=22, color="#666666")
    for month, lab, col, mk, ms, off in zip(MONTHS, MONTH_LABELS, MONTH_COLORS, MONTH_MARKERS,
                                            [17, 16, 15], [-0.12, 0, 0.12]):
        s = r[r.Month == month].set_index("Patient").reindex(PATIENTS)
        ok = ~s.Balanced_Accuracy.isna().values
        x = np.arange(len(PATIENTS)) + off
        ax.errorbar(x[ok], s.Balanced_Accuracy.values[ok], yerr=s.BalAcc_Fold_SD.values[ok],
                    color=col, lw=3, marker=mk, ms=ms, mec="white", mew=2,
                    capsize=7, capthick=2.5, elinewidth=2.5, label=lab, zorder=3)
    ax.set_xticks(range(len(PATIENTS)))
    ax.set_xticklabels(PATIENTS, fontsize=30)
    ax.set_xlabel("Pacientes", fontsize=30)
    ax.set_ylabel("Exactitud balanceada (%)", fontsize=28)
    ax.set_ylim(28, 78)
    ax.set_yticks([30, 40, 50, 60, 70])
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d%%"))
    ax.set_xlim(-0.45, 4.45)
    ax.grid(color="#dcdcdc", lw=1.2)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=26)
    for sp in ax.spines.values():
        sp.set_color("#999999")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=3, frameon=False,
              fontsize=27, handlelength=2.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def plot_confusion_month6(cms, out_path):
    mats = [np.array(v, float) for k, v in cms.items() if k.startswith(f"{MODE}|Month6|")]
    pct = np.array([m / m.sum(axis=1, keepdims=True) * 100 for m in mats])
    mu, sd = pct.mean(0), pct.std(0, ddof=1)
    names = ["Reposo", "10 %", "40 %"]
    base = ["#6A3D9A", "#C00000", "#1F4E9A"]
    fig, ax = plt.subplots(figsize=(9.2, 7.6), dpi=200)
    for i in range(3):
        for j in range(3):
            rgb = np.array(to_rgb(base[i]))
            col = rgb if i == j else 1 - (1 - rgb) * 0.18
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=col, edgecolor="white", lw=4))
            ax.text(j + .5, i + .5, f"{mu[i, j]:.1f} % ± {sd[i, j]:.1f}", ha="center", va="center",
                    fontsize=25, fontweight="bold" if i == j else "normal",
                    color="white" if i == j else "#1a1a1a")
    ax.set_xlim(0, 3)
    ax.set_ylim(3, 0)
    ax.set_aspect("equal")
    ax.set_xticks([.5, 1.5, 2.5])
    ax.set_xticklabels(names, fontsize=27)
    ax.xaxis.tick_top()
    ax.set_yticks([.5, 1.5, 2.5])
    ax.set_yticklabels(names, fontsize=27)
    ax.set_xlabel("Clase predicha", fontsize=28, labelpad=12)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("Clase real", fontsize=28)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print("Month 6 mean diagonal (balanced accuracy, mean of patients): %.2f %%" % np.diag(mu).mean())


def main():
    res = pd.read_csv(os.path.join(cfg.CLASSIFICATION_DIR, "results_W700_intrasubject_per_patient.csv"))
    with open(os.path.join(cfg.CLASSIFICATION_DIR, "cms_W700_intrasubject_LR_3class.json")) as fh:
        cms = json.load(fh)
    os.makedirs(cfg.PLOTS_DIR, exist_ok=True)
    a = os.path.join(cfg.PLOTS_DIR, "poster_fig8a_balanced_accuracy.png")
    b = os.path.join(cfg.PLOTS_DIR, "poster_fig8b_confusion_month6.png")
    plot_accuracy_per_patient(res, a)
    plot_confusion_month6(cms, b)
    print("Saved:", a)
    print("Saved:", b)


if __name__ == "__main__":
    main()
