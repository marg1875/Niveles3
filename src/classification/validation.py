"""
Cross-validation strategies shared across classify_*.py scripts.

- lopo_split(): Leave-One-Patient-Out splitter (inter-subject generalization).
- run_intra_subject_pool_cv(): per-patient StratifiedKFold CV with pooled
  predictions (intra-subject). This exact loop — per-patient split, scaler
  fit per fold, predictions pooled across all patients/folds — was
  previously duplicated near-identically across
  scripts/classify_w700_full.py, classify_windows.py, fast_sweep.py,
  classify_fractal_individual.py, fast_intra.py and (as
  `_classify_patient_cv`) classify_fractal_extended.py, including the same
  `n_splits = min(5, max(2, len(y_p) // 3))` formula inlined in each copy.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def run_intra_subject_pool_cv(X, y, patients, model, random_state=42,
                               min_patient_samples=10, min_splits=2, max_splits=5,
                               samples_per_split_divisor=3):
    """Intra-subject StratifiedKFold CV, pooling predictions across patients/folds.

    For each patient: skip if fewer than `min_patient_samples` epochs or
    only one class present. Otherwise run StratifiedKFold with
    `n_splits = clamp(len(y_p) // samples_per_split_divisor, min_splits, max_splits)`,
    fitting a fresh StandardScaler per fold (train-only) and refitting
    `model` per fold — `model` is reused/mutated in place across all folds
    and patients, standard sklearn usage since each .fit() call fully resets
    the estimator's state.

    Args:
        X: (n_samples, n_features) feature matrix.
        y: (n_samples,) class labels.
        patients: (n_samples,) patient id per row.
        model: sklearn-compatible estimator with .fit()/.predict().
        random_state: passed to StratifiedKFold(shuffle=True, ...).
        min_patient_samples: skip a patient with fewer epochs than this.
        min_splits, max_splits, samples_per_split_divisor: n_splits formula.

    Returns:
        (y_true, y_pred): 1D arrays of pooled true/predicted labels across
        every fold of every patient that had enough data. Empty arrays if
        no patient qualified.
    """
    patients = np.asarray(patients)
    all_yt, all_yp = [], []

    for patient in sorted(set(patients)):
        p_mask = patients == patient
        X_p, y_p = X[p_mask], y[p_mask]
        if len(y_p) < min_patient_samples or len(np.unique(y_p)) < 2:
            continue

        n_splits = min(max_splits, max(min_splits, len(y_p) // samples_per_split_divisor))
        try:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
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

    return np.array(all_yt), np.array(all_yp)


def lopo_split(df):
    """Leave-One-Patient-Out cross-validation generator.

    Args:
        df: DataFrame with "Patient" column.

    Yields:
        (train_idx, test_idx, test_patient) as boolean masks + patient name.
    """
    patients = sorted(df["Patient"].unique())

    for test_patient in patients:
        train_idx = (df["Patient"] != test_patient).values
        test_idx = (df["Patient"] == test_patient).values
        yield train_idx, test_idx, test_patient


def get_patient_counts(df):
    """Return dict of patient -> epoch count."""
    return df["Patient"].value_counts().to_dict()
