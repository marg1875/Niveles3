"""
Fast classification for the multi-window sweep CSVs.

Uses RandomForest on Delta features (no per-channel, no Martinez).
For each window size, runs intra-subject classification across:
- Mixed (all months), Month1, Month3, Month6
- 3-class and 2-class
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

WINDOW_SIZES = list(range(500, 2050, 100))
FEATURES_DIR = cfg.FEATURES_DIR
DELTA_COLS = ["Delta_RS", "Delta_Higuchi", "Delta_DFA", "Delta_Variogram", "Delta_Average"]


def run_intra(df, feature_cols, class_type, month_filter=None):
    """Intra-subject classification for a given data slice."""
    if month_filter:
        df = df[df["Session"] == month_filter].copy()
        if len(df) < 10:
            return None

    pat_col = "Patient"
    y = df["MVR_Class"].values
    pat = df[pat_col].values

    if class_type == "2class":
        mask = np.isin(y, (10, 40))
        y = y[mask]
        pat = pat[mask]
        if len(y) < 10 or len(np.unique(y)) < 2:
            return None

    X = df[feature_cols].values
    if class_type == "2class":
        X = X[mask]

    X = np.nan_to_num(X, nan=0.0)

    model = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
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


def main():
    print("=" * 60)
    print("FAST SWEEP CLASSIFICATION")
    print("=" * 60)
    print(f"Model: RandomForest (300 trees) | Features: Delta (5 fractals)")
    print("=" * 60)

    t0 = time.time()
    all_results = []

    for w in WINDOW_SIZES:
        csv_path = os.path.join(FEATURES_DIR, f"FastSweep_W{w}_Features.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        print(f"\nW{w:4d} ({w/cfg.FS:.1f}s): {len(df)} epochs")

        # Check which feature columns exist
        available = [c for c in DELTA_COLS if c in df.columns]
        if len(available) < 3:
            print(f"  Not enough feature columns: {available}")
            continue

        for month_label, month_val in [("Mixed", None), ("Month1", "Month1"),
                                        ("Month3", "Month3"), ("Month6", "Month6")]:
            for ct in ["3class", "2class"]:
                res = run_intra(df, available, ct, month_val)
                if res:
                    res["Window_Samples"] = w
                    res["Window_Sec"] = w / cfg.FS
                    res["Month"] = month_label
                    res["Class_Type"] = ct
                    res["N_Features"] = len(available)
                    all_results.append(res)
                    print(f"  {month_label:8s} {ct:6s}: Acc={res['Accuracy']:.4f} F1={res['F1']:.4f} N={res['N']}")

    if not all_results:
        print("No results. Run fast_sweep.py first.")
        return

    df_r = pd.DataFrame(all_results)
    out_path = os.path.join(cfg.CLASSIFICATION_DIR, "results_sweep_fast.csv")
    df_r.to_csv(out_path, index=False)

    print("\n" + "=" * 60)
    print("BEST BY CONDITION (3-class)")
    print("=" * 60)
    for month in ["Mixed", "Month1", "Month3", "Month6"]:
        sub = df_r[(df_r["Class_Type"] == "3class") & (df_r["Month"] == month)]
        if len(sub) == 0:
            continue
        best = sub.loc[sub["Accuracy"].idxmax()]
        print(f"\n{month}: W{best['Window_Samples']:4d} ({best['Window_Sec']:.1f}s) "
              f"Acc={best['Accuracy']:.4f} F1={best['F1']:.4f} N={best['N']}")
        for _, r in sub.nlargest(5, "Accuracy").iterrows():
            print(f"  W{r['Window_Samples']:4d} ({r['Window_Sec']:.1f}s) Acc={r['Accuracy']:.4f} F1={r['F1']:.4f}")

    print("\n" + "=" * 60)
    print("BEST BY CONDITION (2-class)")
    print("=" * 60)
    for month in ["Mixed", "Month1", "Month3", "Month6"]:
        sub = df_r[(df_r["Class_Type"] == "2class") & (df_r["Month"] == month)]
        if len(sub) == 0:
            continue
        best = sub.loc[sub["Accuracy"].idxmax()]
        print(f"\n{month}: W{best['Window_Samples']:4d} ({best['Window_Sec']:.1f}s) "
              f"Acc={best['Accuracy']:.4f} F1={best['F1']:.4f}")

    # Overall best window (average 3-class for Mixed + Month1 + Month3)
    print("\n" + "=" * 60)
    print("OVERALL BEST WINDOW (avg 3-class Mixed+Month1+Month3)")
    print("=" * 60)
    mixed_3c = df_r[(df_r["Class_Type"] == "3class") & (df_r["Month"] == "Mixed")]
    m1_3c = df_r[(df_r["Class_Type"] == "3class") & (df_r["Month"] == "Month1")]
    m3_3c = df_r[(df_r["Class_Type"] == "3class") & (df_r["Month"] == "Month3")]

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
        print(f"\nBEST WINDOW: W{best_w} ({best_w/cfg.FS:.1f}s) avg acc={avg_by_window[best_w]:.4f}")
        print(f"\nTop 5 windows:")
        for w, avg in sorted(avg_by_window.items(), key=lambda x: -x[1])[:5]:
            print(f"  W{w:4d} ({w/cfg.FS:.1f}s): avg={avg:.4f}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. Results: {out_path}")


if __name__ == "__main__":
    main()
