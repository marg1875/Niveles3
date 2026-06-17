"""Full classification for best window W700 (2.8s)."""
import os, sys, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from src.classification.features import get_channel_subset_features, get_feature_subset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score

FEATURES_DIR = cfg.FEATURES_DIR
RESULTS_DIR = cfg.CLASSIFICATION_DIR

MODELS = {
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "SVM": SVC(random_state=42, probability=True),
    "LogReg": LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1),
    "kNN": KNeighborsClassifier(n_neighbors=10, metric="chebyshev", n_jobs=-1),
    "NaiveBayes": GaussianNB(),
}

FEATURE_SUBSETS = ["Delta", "Delta+Martinez", "Martinez", "Delta+Spectral"]
CHANNEL_CONF = "All-16"
CLASS_TYPES = ["3class", "2class"]
MONTH_CONDITIONS = [("Mixed", None), ("Month1", "Month1"), ("Month3", "Month3"), ("Month6", "Month6")]


def classify_condition(df_raw, feat_subset, mname, model, class_type, month_filter):
    df_ch = get_channel_subset_features(df_raw, CHANNEL_CONF)
    if month_filter:
        df_ch = df_ch[df_ch["Session"] == month_filter].copy()

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

    all_yt, all_yp = [], []
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
            all_yt.extend(y_te)
            all_yp.extend(y_pred)

    if len(all_yt) > 0:
        acc = accuracy_score(all_yt, all_yp)
        f1 = f1_score(all_yt, all_yp, average="macro", zero_division=0)
        return {"Accuracy": acc, "F1": f1, "N": len(all_yt)}
    return None


def main():
    csv_path = os.path.join(FEATURES_DIR, "Fixed_W700_Full_Features.csv")
    if not os.path.exists(csv_path):
        print("CSV not found:", csv_path)
        return

    print("=" * 60)
    print("FULL CLASSIFICATION W700 (2.8s) - Intra-Subject")
    print("=" * 60)
    print("CSV:", csv_path)
    print("Models:", list(MODELS.keys()))
    print("Feature subsets:", FEATURE_SUBSETS)
    print("Channel:", CHANNEL_CONF)
    print("=" * 60)

    t0 = time.time()
    df_raw = pd.read_csv(csv_path)
    print("Loaded {} epochs".format(len(df_raw)))

    results = []
    total = len(MODELS) * len(FEATURE_SUBSETS) * len(CLASS_TYPES) * len(MONTH_CONDITIONS)
    done = 0

    for month_label, month_val in MONTH_CONDITIONS:
        for feat_subset in FEATURE_SUBSETS:
            for ct in CLASS_TYPES:
                for mname, model in MODELS.items():
                    done += 1
                    res = classify_condition(df_raw, feat_subset, mname, model, ct, month_val)
                    if res:
                        res["Model"] = mname
                        res["Feature_Subset"] = feat_subset
                        res["Class_Type"] = ct
                        res["Month"] = month_label
                        results.append(res)
                        print("[{}/{}] {:20s} {:16s} {:8s} {:6s}: Acc={:.4f} F1={:.4f} N={}".format(
                            done, total, mname, feat_subset, month_label, ct,
                            res["Accuracy"], res["F1"], res["N"]))
                    else:
                        print("[{}/{}] {:20s} {:16s} {:8s} {:6s}: SKIP".format(
                            done, total, mname, feat_subset, month_label, ct))

    df_r = pd.DataFrame(results)
    out_path = os.path.join(RESULTS_DIR, "results_W700_full_intra.csv")
    df_r.to_csv(out_path, index=False)

    print("\n" + "=" * 60)
    print("BEST RESULTS BY MONTH (2-class)")
    print("=" * 60)
    for month in ["Mixed", "Month1", "Month3", "Month6"]:
        sub = df_r[(df_r["Class_Type"] == "2class") & (df_r["Month"] == month)]
        if len(sub) == 0:
            continue
        best = sub.loc[sub["Accuracy"].idxmax()]
        print("\n{}: {} + {} = {:.4f} (F1={:.4f})".format(
            month, best["Model"], best["Feature_Subset"], best["Accuracy"], best["F1"]))
        for _, r in sub.nlargest(5, "Accuracy").iterrows():
            print("  {:20s} {:16s} Acc={:.4f} F1={:.4f}".format(r["Model"], r["Feature_Subset"], r["Accuracy"], r["F1"]))

    elapsed = time.time() - t0
    print("\nDone in {:.0f}s. Results: {}".format(elapsed, out_path))


if __name__ == "__main__":
    main()
