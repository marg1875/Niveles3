"""
Quick multi-window classification for Niveles3.

For each fixed-window CSV:
- RandomForest + Delta+Martinez + All-16 channels
- 4 conditions: Mixed, Month1, Month3, Month6
- Intra-subject 5-fold CV pooled across patients

Finds the optimal window size and runs per-month analysis.
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from src.classification.features import get_channel_subset_features, get_feature_subset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

WINDOW_SIZES = list(range(500, 2050, 100))
FEATURES_DIR = cfg.FEATURES_DIR
RESULTS_DIR = cfg.CLASSIFICATION_DIR


def run_intra_experiment(df_all, ch_conf, feat_subset, class_type, month_filter=None):
    """Run intra-subject classification for a specific condition.

    Args:
        df_all: Full feature DataFrame.
        ch_conf: Channel config name (e.g. "All-16").
        feat_subset: Feature subset name (e.g. "Delta+Martinez").
        class_type: "3class" or "2class".
        month_filter: "Month1", "Month3", "Month6", or None for Mixed.

    Returns:
        dict with Accuracy, F1, N, or None if failed.
    """
    df_ch = get_channel_subset_features(df_all, ch_conf)

    if month_filter:
        df_ch = df_ch[df_ch["Session"] == month_filter].copy()
        if len(df_ch) < 10:
            return None

    try:
        X, y, feat_cols, df_filt = get_feature_subset(df_ch, feat_subset)
    except Exception:
        return None

    if len(feat_cols) < 2 or len(y) < 10:
        return None

    if class_type == "2class":
        mask = np.isin(y, (10, 40))
        X = X[mask]
        y = y[mask]
        pat = df_filt.iloc[mask]["Patient"].values
    else:
        pat = df_filt["Patient"].values

    if len(y) < 10 or len(np.unique(y)) < 2:
        return None

    model = RandomForestClassifier(n_estimators=300, max_depth=None,
                                   random_state=42, n_jobs=-1)
    all_y_true, all_y_pred = [], []

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
        for tr, te in skf.split(X_p, y_p):
            X_tr, X_te = X_p[tr], X_p[te]
            y_tr, y_te = y_p[tr], y_p[te]
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            model.fit(X_tr_s, y_tr)
            y_pred = model.predict(X_te_s)
            all_y_true.extend(y_te)
            all_y_pred.extend(y_pred)

    if len(all_y_true) > 0:
        acc = accuracy_score(all_y_true, all_y_pred)
        f1 = f1_score(all_y_true, all_y_pred, average="macro", zero_division=0)
        return {"Accuracy": acc, "F1": f1, "N": len(all_y_true)}
    return None


def classify_window(csv_path, w_samples):
    """Classify one window size across all conditions."""
    if not os.path.exists(csv_path):
        print(f"  W{w_samples}: CSV not found")
        return []

    df = pd.read_csv(csv_path)
    print(f"  W{w_samples} ({w_samples/cfg.FS:.1f}s): {len(df)} epochs loaded")

    results = []
    for month_label, month_val in [("Mixed", None), ("Month1", "Month1"),
                                    ("Month3", "Month3"), ("Month6", "Month6")]:
        for ct in ["3class", "2class"]:
            res = run_intra_experiment(df, "All-16", "Delta+Martinez", ct, month_val)
            if res:
                res["Window_Samples"] = w_samples
                res["Window_Sec"] = w_samples / cfg.FS
                res["Month"] = month_label
                res["Class_Type"] = ct
                results.append(res)
                print(f"    {month_label:8s} {ct:6s}: Acc={res['Accuracy']:.4f} F1={res['F1']:.4f} N={res['N']}")
            else:
                print(f"    {month_label:8s} {ct:6s}: SKIP (insufficient data)")

    return results


def main():
    print("=" * 60)
    print("MULTI-WINDOW QUICK CLASSIFICATION")
    print("=" * 60)
    print(f"Model: RandomForest (n_estimators=300)")
    print(f"Features: Delta+Martinez | Channels: All-16")
    print(f"Validation: Intra-subject 5-fold CV")
    print("=" * 60)

    t0 = time.time()
    all_results = []

    window_order = []
    for w in WINDOW_SIZES:
        csv_path = os.path.join(FEATURES_DIR, f"Fixed_W{w}_Features.csv")
        if os.path.exists(csv_path):
            window_order.append(w)
            res_list = classify_window(csv_path, w)
            all_results.extend(res_list)

    if not all_results:
        print("No results found. Run extraction first.")
        return

    df_r = pd.DataFrame(all_results)
    out_path = os.path.join(RESULTS_DIR, "results_windows_quick.csv")
    df_r.to_csv(out_path, index=False)

    print("\n" + "=" * 60)
    print("BEST BY CONDITION (3-class)")
    print("=" * 60)
    for month in ["Mixed", "Month1", "Month3", "Month6"]:
        sub = df_r[(df_r["Class_Type"] == "3class") & (df_r["Month"] == month)]
        if len(sub) == 0:
            print(f"\n{month}: No results")
            continue
        best = sub.loc[sub["Accuracy"].idxmax()]
        print(f"\n{month}: W{best['Window_Samples']:4d} ({best['Window_Sec']:.1f}s) "
              f"Acc={best['Accuracy']:.4f} F1={best['F1']:.4f} N={best['N']}")
        top3 = sub.nlargest(3, "Accuracy")
        for _, r in top3.iterrows():
            print(f"  W{r['Window_Samples']:4d} ({r['Window_Sec']:.1f}s) "
                  f"Acc={r['Accuracy']:.4f} F1={r['F1']:.4f}")

    print("\n" + "=" * 60)
    print("BEST BY CONDITION (2-class)")
    print("=" * 60)
    for month in ["Mixed", "Month1", "Month3", "Month6"]:
        sub = df_r[(df_r["Class_Type"] == "2class") & (df_r["Month"] == month)]
        if len(sub) == 0:
            continue
        best = sub.loc[sub["Accuracy"].idxmax()]
        print(f"\n{month}: W{best['Window_Samples']:4d} ({best['Window_Sec']:.1f}s) "
              f"Acc={best['Accuracy']:.4f} F1={best['F1']:.4f} N={best['N']}")
        top3 = sub.nlargest(3, "Accuracy")
        for _, r in top3.iterrows():
            print(f"  W{r['Window_Samples']:4d} ({r['Window_Sec']:.1f}s) "
                  f"Acc={r['Accuracy']:.4f} F1={r['F1']:.4f}")

    # Find overall best window (average of 3-class Mixed + Month1 + Month3)
    print("\n" + "=" * 60)
    print("OVERALL BEST WINDOW (avg 3-class Mixed+Month1+Month3)")
    print("=" * 60)
    mixed_3c = df_r[(df_r["Class_Type"] == "3class") & (df_r["Month"] == "Mixed")].copy()
    m1_3c = df_r[(df_r["Class_Type"] == "3class") & (df_r["Month"] == "Month1")].copy()
    m3_3c = df_r[(df_r["Class_Type"] == "3class") & (df_r["Month"] == "Month3")].copy()

    if len(mixed_3c) > 0:
        avg_by_window = {}
        for _, row in mixed_3c.iterrows():
            w = row["Window_Samples"]
            vals = [row["Accuracy"]]
            m1 = m1_3c[m1_3c["Window_Samples"] == w]
            m3 = m3_3c[m3_3c["Window_Samples"] == w]
            if len(m1) > 0:
                vals.append(m1.iloc[0]["Accuracy"])
            if len(m3) > 0:
                vals.append(m3.iloc[0]["Accuracy"])
            avg_by_window[w] = np.mean(vals)

        best_w = max(avg_by_window, key=avg_by_window.get)
        print(f"\nBest window: W{best_w} ({best_w/cfg.FS:.1f}s) avg acc={avg_by_window[best_w]:.4f}")
        print(f"\nAll windows sorted:")
        for w, avg in sorted(avg_by_window.items(), key=lambda x: -x[1])[:5]:
            print(f"  W{w:4d} ({w/cfg.FS:.1f}s): avg={avg:.4f}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. Results: {out_path}")


if __name__ == "__main__":
    main()
