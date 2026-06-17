"""
LOPO (Leave-One-Patient-Out) cross-validation splitter.

Yields (train_indices, test_indices, test_patient) for each fold.
"""
import numpy as np


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
