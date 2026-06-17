import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve, auc
import config as cfg

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.25,
    "grid.linestyle": ":",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

MODEL_COLORS = {
    "SVM": "#E74C3C",
    "kNN": "#3498DB",
    "MLP": "#9B59B6",
    "RandomForest": "#2ECC71",
    "NaiveBayes": "#1ABC9C",
    "BayesNet": "#E67E22",
    "RandomTree": "#95A5A6",
    "LDA": "#F39C12",
    "LogisticRegression": "#34495E",
}

FEATURE_COLORS = {
    "Delta": "#1A5276",
    "Martinez": "#E74C3C",
    "Delta+Martinez": "#27AE60",
    "Delta+Spectral": "#8E44AD",
}

_EXTRA_MODEL_COLORS = [
    "#D35400", "#2874A6", "#7D3C98", "#1E8449", "#117A65",
    "#B03A2E", "#6C3483", "#F1C40F", "#148F77", "#BA4A00",
]
_extra_idx = 0


def _get_color(model_name):
    if model_name in MODEL_COLORS:
        return MODEL_COLORS[model_name]
    global _extra_idx
    color = _EXTRA_MODEL_COLORS[_extra_idx % len(_EXTRA_MODEL_COLORS)]
    _extra_idx += 1
    return color


def _ensure_output_dir(output_dir):
    if output_dir is None:
        output_dir = cfg.PLOTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _parse_confusion_matrix(cm_str):
    if not isinstance(cm_str, str):
        return None
    cleaned = re.sub(r"\s+", " ", cm_str.strip())
    cleaned = cleaned.replace("[ ", "[").replace(" ]", "]")
    rows = re.findall(r"\[([0-9\s]+)\]", cleaned)
    if not rows:
        return None
    matrix = []
    for row_str in rows:
        vals = [int(x) for x in row_str.strip().split()]
        if vals:
            matrix.append(vals)
    return np.array(matrix)


def _find_best_per_config(df_3class, results_csv_path):
    idx_best = df_3class.groupby("Channel_Config")["accuracy_mean"].idxmax()
    return df_3class.loc[idx_best].copy()


def plot_ranking(results_csv_path, output_dir=None):
    if not os.path.isfile(results_csv_path):
        print(f"Warning: results file not found — {results_csv_path}")
        return
    output_dir = _ensure_output_dir(output_dir)

    df = pd.read_csv(results_csv_path)
    mask = (df["Class_Type"] == "3class") & df["accuracy_mean"].notna()
    df3 = df[mask].copy()
    if df3.empty:
        print("Warning: no 3class results with valid accuracy found.")
        return
    df3 = df3.sort_values("accuracy_mean", ascending=False).head(20)
    df3 = df3.iloc[::-1]

    labels = [f"{r.Model} | {r.Channel_Config} | {r.Feature_Subset}"
              for _, r in df3.iterrows()]
    colors = [_get_color(m) for m in df3["Model"]]
    accuracies = df3["accuracy_mean"].values

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(labels)), accuracies, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Accuracy")
    ax.set_title("Top 20 Experiments — 3-Class LOPO")
    ax.set_xlim(0, max(accuracies) * 1.12)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{acc:.3f}", va="center", fontsize=7)

    legend_handles = []
    seen = set()
    for m in df3["Model"]:
        if m not in seen:
            seen.add(m)
            legend_handles.append(Patch(color=_get_color(m), label=m))
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, ncol=2)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "ranking_top20_3class.png"))
    plt.close(fig)


def plot_accuracy_vs_channels(results_csv_path, output_dir=None):
    if not os.path.isfile(results_csv_path):
        print(f"Warning: results file not found — {results_csv_path}")
        return
    output_dir = _ensure_output_dir(output_dir)

    df = pd.read_csv(results_csv_path)
    mask = (df["Class_Type"] == "3class") & df["accuracy_mean"].notna()
    df3 = df[mask].copy()
    if df3.empty:
        print("Warning: no 3class results with valid accuracy found.")
        return

    target_n = [1, 2, 4, 6, 16]
    df3 = df3[df3["N_Channels"].isin(target_n)]

    grouped = df3.groupby(["Feature_Subset", "N_Channels"])["accuracy_mean"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    feature_subsets = sorted(grouped["Feature_Subset"].unique(),
                             key=lambda x: df3[df3["Feature_Subset"] == x]["accuracy_mean"].mean(),
                             reverse=True)

    for fs in feature_subsets:
        sub = grouped[grouped["Feature_Subset"] == fs]
        color = FEATURE_COLORS.get(fs, "#7F8C8D")
        ax.plot(sub["N_Channels"], sub["mean"], marker="o", color=color,
                linewidth=2, markersize=7, label=fs)
        ax.fill_between(sub["N_Channels"],
                        sub["mean"] - sub["std"],
                        sub["mean"] + sub["std"],
                        color=color, alpha=0.12)

    ax.set_xlabel("Number of Channels")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Number of Channels (3-Class)")
    ax.set_xticks(target_n)
    ax.set_xticklabels([str(n) for n in target_n])
    ax.grid(True)
    ax.legend(fontsize=9, loc="best")
    ax.set_ylim(0.35, 0.85)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "accuracy_vs_channels.png"))
    plt.close(fig)


def plot_confusion_matrices(roc_data_dir, results_csv_path, output_dir=None):
    if not os.path.isfile(results_csv_path):
        print(f"Warning: results file not found — {results_csv_path}")
        return
    output_dir = _ensure_output_dir(output_dir)

    df = pd.read_csv(results_csv_path)
    mask = (df["Class_Type"] == "3class") & df["accuracy_mean"].notna()
    df3 = df[mask].copy()
    if df3.empty:
        print("Warning: no 3class results with valid accuracy found.")
        return

    best_per_config = _find_best_per_config(df3, results_csv_path)
    class_labels = ["cls0", "cls10", "cls40"]

    for _, row in best_per_config.iterrows():
        channel = row["Channel_Config"]
        cm_str = row.get("confusion_matrix_sum", None)
        cm = _parse_confusion_matrix(cm_str) if pd.notna(cm_str) else None

        if cm is None:
            print(f"Warning: no confusion matrix for {channel}, skipping.")
            continue

        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

        fig, ax = plt.subplots(figsize=(6, 5.5))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                pct = f"{cm_norm[i, j]:.1%}" if cm_norm[i, j] < 0.995 else "100%"
                text_color = "white" if cm_norm[i, j] > 0.55 else "black"
                ax.text(j, i, f"{cm[i, j]}\n({pct})", ha="center", va="center",
                        fontsize=10, color=text_color, fontweight="bold")

        display_labels = class_labels[:cm.shape[0]]
        ax.set_xticks(range(len(display_labels)))
        ax.set_xticklabels(display_labels)
        ax.set_yticks(range(len(display_labels)))
        ax.set_yticklabels(display_labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        model = row["Model"]
        fs = row["Feature_Subset"]
        acc = row["accuracy_mean"]
        ax.set_title(f"Confusion Matrix — {channel}\n{model} | {fs} | Acc={acc:.3f}")

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Proportion")

        plt.tight_layout()
        safe_name = channel.replace(" ", "_").replace("/", "_")
        fig.savefig(os.path.join(output_dir, f"confusion_matrix_{safe_name}.png"))
        plt.close(fig)


def plot_roc_curves(roc_data_dir, results_csv_path, output_dir=None):
    if not os.path.isfile(results_csv_path):
        print(f"Warning: results file not found — {results_csv_path}")
        return
    output_dir = _ensure_output_dir(output_dir)

    df = pd.read_csv(results_csv_path)

    mask_3 = (df["Class_Type"] == "3class") & df["accuracy_mean"].notna()
    df3 = df[mask_3].copy()
    mask_2 = (df["Class_Type"] == "2class") & df["accuracy_mean"].notna()
    df2 = df[mask_2].copy()

    top3_3class = df3.sort_values("accuracy_mean", ascending=False).head(3) if not df3.empty else pd.DataFrame()
    top3_2class = df2.sort_values("accuracy_mean", ascending=False).head(3) if not df2.empty else pd.DataFrame()

    has_3class = not top3_3class.empty and os.path.isdir(roc_data_dir)
    has_2class = not top3_2class.empty and os.path.isdir(roc_data_dir)

    if not has_3class and not has_2class:
        print("Warning: no ROC data available.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    if has_3class:
        ax = axes[0]
        for _, row in top3_3class.iterrows():
            model = row["Model"]
            channel = row["Channel_Config"]
            fs = row["Feature_Subset"]
            label = f"{model} | {channel} | {fs}"
            npz_name = f"{model}_{channel}_{fs}.npz"
            npz_path = os.path.join(roc_data_dir, npz_name)

            if os.path.isfile(npz_path):
                data = np.load(npz_path, allow_pickle=True)
                y_true = data.get("y_true", None)
                y_score = data.get("y_score", None)
                if y_true is not None and y_score is not None:
                    n_classes = y_score.shape[1]
                    fpr, tpr, roc_auc = {}, {}, {}
                    for i in range(n_classes):
                        y_bin = (y_true == i).astype(int)
                        fpr[i], tpr[i], _ = roc_curve(y_bin, y_score[:, i])
                        roc_auc[i] = auc(fpr[i], tpr[i])
                    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
                    mean_tpr = np.zeros_like(all_fpr)
                    for i in range(n_classes):
                        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
                    mean_tpr /= n_classes
                    macro_auc = auc(all_fpr, mean_tpr)
                    ax.plot(all_fpr, mean_tpr, linewidth=2,
                            label=f"{label} (macro AUC={macro_auc:.3f})")
                data.close()
            else:
                ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, alpha=0.5,
                        label=f"{label} (.npz missing)")

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves — Best 3-Class Experiments (Macro Avg)")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True)
    else:
        axes[0].text(0.5, 0.5, "No 3-class ROC data", ha="center", va="center",
                     transform=axes[0].transAxes, fontsize=14, color="gray")
        axes[0].set_title("3-Class ROC")

    if has_2class:
        ax = axes[1]
        for _, row in top3_2class.iterrows():
            model = row["Model"]
            channel = row["Channel_Config"]
            fs = row["Feature_Subset"]
            label = f"{model} | {channel} | {fs}"
            npz_name = f"{model}_{channel}_{fs}.npz"
            npz_path = os.path.join(roc_data_dir, npz_name)

            if os.path.isfile(npz_path):
                data = np.load(npz_path, allow_pickle=True)
                y_true = data.get("y_true", None)
                y_score = data.get("y_score", None)
                if y_true is not None and y_score is not None:
                    if y_score.ndim > 1 and y_score.shape[1] > 1:
                        proba = y_score[:, 1]
                    else:
                        proba = y_score.ravel()
                    fpr, tpr, _ = roc_curve(y_true, proba)
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, linewidth=2,
                            label=f"{label} (AUC={roc_auc:.3f})")
                data.close()
            else:
                ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, alpha=0.5,
                        label=f"{label} (.npz missing)")

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves — Best 2-Class Experiments")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True)
    else:
        axes[1].text(0.5, 0.5, "No 2-class ROC data", ha="center", va="center",
                     transform=axes[1].transAxes, fontsize=14, color="gray")
        axes[1].set_title("2-Class ROC")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "roc_curves_best.png"))
    plt.close(fig)


def plot_model_comparison(results_csv_path, output_dir=None):
    if not os.path.isfile(results_csv_path):
        print(f"Warning: results file not found — {results_csv_path}")
        return
    output_dir = _ensure_output_dir(output_dir)

    df = pd.read_csv(results_csv_path)
    mask_3 = (df["Class_Type"] == "3class") & df["accuracy_mean"].notna()
    mask_2 = (df["Class_Type"] == "2class") & df["accuracy_mean"].notna()

    def _group_models(sub_df, class_label):
        grouped = sub_df.groupby("Model")["accuracy_mean"].agg(["mean", "std"]).reset_index()
        grouped = grouped.sort_values("mean", ascending=True)
        return grouped

    def _plot_panel(ax, grouped, title):
        models = grouped["Model"].tolist()
        means = grouped["mean"].values
        stds = grouped["std"].values
        colors = [_get_color(m) for m in models]

        bars = ax.barh(range(len(models)), means, xerr=stds, color=colors,
                       edgecolor="white", linewidth=0.8, capsize=3)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=10)
        ax.set_xlabel("Accuracy")
        ax.set_title(title)
        ax.set_xlim(0, max(means + stds) * 1.18)

        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{mean:.3f}", va="center", fontsize=8)

    grouped_3 = _group_models(df[mask_3], "3class") if mask_3.any() else pd.DataFrame()
    grouped_2 = _group_models(df[mask_2], "2class") if mask_2.any() else pd.DataFrame()

    if grouped_3.empty and grouped_2.empty:
        print("Warning: no model data to compare.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    if not grouped_3.empty:
        _plot_panel(axes[0], grouped_3, "3-Class — Mean Accuracy by Model")
    else:
        axes[0].text(0.5, 0.5, "No 3-class data", ha="center", va="center",
                     transform=axes[0].transAxes, fontsize=14, color="gray")
        axes[0].set_title("3-Class")

    if not grouped_2.empty:
        _plot_panel(axes[1], grouped_2, "2-Class — Mean Accuracy by Model")
    else:
        axes[1].text(0.5, 0.5, "No 2-class data", ha="center", va="center",
                     transform=axes[1].transAxes, fontsize=14, color="gray")
        axes[1].set_title("2-Class")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "model_ranking_combined.png"))
    plt.close(fig)


def plot_recall_per_class(results_csv_path, output_dir=None):
    if not os.path.isfile(results_csv_path):
        print(f"Warning: results file not found — {results_csv_path}")
        return
    output_dir = _ensure_output_dir(output_dir)

    df = pd.read_csv(results_csv_path)
    mask = (df["Class_Type"] == "3class") & df["accuracy_mean"].notna()
    df3 = df[mask].copy()
    if df3.empty:
        print("Warning: no 3class results with valid accuracy found.")
        return

    best_per_config = _find_best_per_config(df3, results_csv_path)
    configs = best_per_config["Channel_Config"].tolist()
    recall_0 = best_per_config["recall_cls0_mean"].values
    recall_10 = best_per_config["recall_cls10_mean"].values
    recall_40 = best_per_config["recall_cls40_mean"].values

    x = np.arange(len(configs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars0 = ax.bar(x - width, recall_0, width, label="cls0 (rest)", color="#27AE60", edgecolor="white")
    bars10 = ax.bar(x, recall_10, width, label="cls10 (10%)", color="#3498DB", edgecolor="white")
    bars40 = ax.bar(x + width, recall_40, width, label="cls40 (40%)", color="#E74C3C", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("Recall")
    ax.set_title("Recall per Class — Best Experiment per Channel Config (3-Class)")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10)
    ax.grid(axis="y")

    for bars in [bars0, bars10, bars40]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015,
                    f"{h:.2f}", ha="center", fontsize=7, fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "recall_per_class.png"))
    plt.close(fig)
