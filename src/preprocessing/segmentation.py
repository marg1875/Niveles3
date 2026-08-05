"""
Fixed-window segmentation for preprocessed EEG.

Previously duplicated near-verbatim (`for s in range(0, n_samples - W + 1, W):
segments.append(...)`) in scripts/extract_w700_full.py, extract_features_fixed.py,
extract_multi_window.py and fast_sweep.py.
"""
from typing import List, Tuple

import numpy as np


def segment_fixed_window(eeg: np.ndarray, window: int,
                          overlap: float = 0.0) -> Tuple[List[np.ndarray], List[int]]:
    """Split preprocessed EEG into contiguous fixed-size windows.

    Args:
        eeg: (n_channels, n_samples) preprocessed EEG.
        window: Window size in samples (e.g. 700 for W700 = 2.8s @ 250Hz).
        overlap: Fraction of overlap between consecutive windows, in [0, 1).
            0.0 (default) matches every window-sweep script in this repo, all
            of which use non-overlapping contiguous segments.

    Returns:
        (segments, starts): `segments` is a list of (n_channels, window)
        arrays; `starts` is the list of starting sample indices, same length.
        Any trailing samples that don't fill a full window are discarded
        (matching the original inline implementations).
    """
    if not (0.0 <= overlap < 1.0):
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    step = window if overlap == 0.0 else max(1, int(round(window * (1 - overlap))))
    n_samples = eeg.shape[1]

    segments = []
    starts = []
    for s in range(0, n_samples - window + 1, step):
        segments.append(eeg[:, s:s + window])
        starts.append(s)

    return segments, starts
