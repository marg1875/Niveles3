"""Full classification for best window W700 (2.8s)."""
import os, sys, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from joblib import Parallel, delayed
from src.classification.features import get_channel_subset_features, get_feature_subset
from src.classification.validation import run_intra_subject_pool_cv
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score

FEATURES_DIR = cfg.FEATURES_DIR
RESULTS_DIR = cfg.CLASSIFICATION_DIR

# n_jobs=1 on the individual estimators: parallelism now happens one level up
# (across the 160 experiment cells, via cfg.N_JOBS in main()). Leaving these
# at n_jobs=-1 too would oversubscribe CPU cores (N_JOBS worker processes x
# each spawning its own internal thread pool).
MODELS = {
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=1),
    "SVM": SVC(random_state=42, probability=True),
    "LogReg": LogisticRegression(max_iter=2000, random_state=42, n_jobs=1),
    "kNN": KNeighborsClassifier(n_neighbors=10, metric="chebyshev", n_jobs=1),
    "NaiveBayes": GaussianNB(),
}

FEATURE_SUBSETS = ["Delta", "Delta+Martinez", "Martinez", "Spectral", "Delta+Spectral"]
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

    all_yt, all_yp = run_intra_subject_pool_cv(X, y, pat, model)

    if len(all_yt) > 0:
        acc = accuracy_score(all_yt, all_yp)
        f1 = f1_score(all_yt, all_yp, average="macro", zero_division=0)
        return {"Accuracy": acc, "F1": f1, "N": len(all_yt)}
    return None


def main():
    # NOTE: previously read "Fixed_W700_Full_Features.csv", a stale manual
    # copy that scripts/extract_w700_full.py does not write (it writes
    # "Fixed_W700_3class_Features.csv") — re-running extraction silently did
    # not update what this script consumed. Fixed here (moved up from the
    # Fase 3 correctness pass) because Fase 1's end-to-end verification
    # requires the classifier to read the extractor's actual output.
    csv_path = os.path.join(FEATURES_DIR, "Fixed_W700_3class_Features.csv")
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

    # Clean NaN/inf once for the whole numeric matrix here, instead of
    # relying solely on get_feature_subset's per-call np.nan_to_num, which
    # would otherwise re-scan the same columns on every one of the ~160
    # (model x subset x class_type x month) experiment cells below. This is
    # a redundant safety net (get_feature_subset still cleans defensively
    # for other callers) but means every classify_condition() call in this
    # script operates on already-clean data.
    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns
    df_raw[numeric_cols] = np.nan_to_num(
        df_raw[numeric_cols].values, nan=0.0, posinf=0.0, neginf=0.0
    )

    # Build the full (month x subset x class_type x model) experiment grid —
    # each cell is an independent intra-subject CV run, embarrassingly
    # parallel across cfg.N_JOBS worker processes instead of one sequential
    # Python loop (previously 160 cells run one at a time).
    conditions = [
        (month_label, month_val, feat_subset, ct, mname)
        for month_label, month_val in MONTH_CONDITIONS
        for feat_subset in FEATURE_SUBSETS
        for ct in CLASS_TYPES
        for mname in MODELS
    ]
    total = len(conditions)
    print("Running {} experiment cells on {} workers...".format(total, cfg.N_JOBS))

    raw_results = Parallel(n_jobs=cfg.N_JOBS, verbose=5)(
        delayed(classify_condition)(df_raw, feat_subset, mname, MODELS[mname], ct, month_val)
        for month_label, month_val, feat_subset, ct, mname in conditions
    )

    results = []
    for (month_label, month_val, feat_subset, ct, mname), res in zip(conditions, raw_results):
        if res:
            res["Model"] = mname
            res["Feature_Subset"] = feat_subset
            res["Class_Type"] = ct
            res["Month"] = month_label
            results.append(res)
            print("{:20s} {:16s} {:8s} {:6s}: Acc={:.4f} F1={:.4f} N={}".format(
                mname, feat_subset, month_label, ct, res["Accuracy"], res["F1"], res["N"]))
        else:
            print("{:20s} {:16s} {:8s} {:6s}: SKIP".format(mname, feat_subset, month_label, ct))

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
