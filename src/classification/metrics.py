"""
Classification metrics for multi-class and binary classification.

Computes: accuracy, F1-macro, per-class precision/recall, confusion matrix.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score,
)
import warnings


def compute_metrics(y_true, y_pred, y_proba=None, classes=None):
    """Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities (optional, for ROC-AUC).
        classes: List of class labels in order.

    Returns:
        Dict with accuracy, f1_macro, and per-class precision/recall/f1.
    """
    if classes is None:
        classes = sorted(np.unique(y_true))

    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    result = {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
    }

    for i, cls in enumerate(classes):
        result[f"precision_cls{cls}"] = float(precision[i])
        result[f"recall_cls{cls}"] = float(recall[i])
        result[f"f1_cls{cls}"] = float(f1_per_class[i])
        result[f"support_cls{cls}"] = int(support[i])

    result["confusion_matrix"] = cm

    # ROC-AUC (OvR for multi-class)
    if y_proba is not None and len(classes) > 2:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                auc = roc_auc_score(y_true, y_proba, multi_class="ovr",
                                    average="macro")
                result["roc_auc_macro"] = float(auc)
        except Exception:
            result["roc_auc_macro"] = np.nan

    return result


def aggregate_lopo_results(fold_results, classes):
    """Aggregate per-fold LOPO results into means and stds.

    Args:
        fold_results: List of dicts from compute_metrics() for each fold.
        classes: List of class labels.

    Returns:
        Dict with mean and std for each metric across folds.
    """
    aggregated = {}

    for key in fold_results[0].keys():
        if key == "confusion_matrix":
            # Sum confusion matrices across folds
            stacked = np.stack([r[key] for r in fold_results])
            aggregated["confusion_matrix_sum"] = np.sum(stacked, axis=0)
            continue

        values = [r[key] for r in fold_results if not np.isnan(r[key])]
        if len(values) > 0:
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        else:
            aggregated[f"{key}_mean"] = np.nan
            aggregated[f"{key}_std"] = np.nan

    return aggregated
