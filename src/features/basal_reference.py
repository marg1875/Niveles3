"""
Basal (Δ-reference) feature computation for fixed-window pipelines.

Consolidates a block previously duplicated near-verbatim across
scripts/extract_w700_full.py, extract_features_fixed.py (x2, one per
strategy) and extract_multi_window.py: average the fractal features of the
first `n_basal` segments of a file into a single basal reference used to
compute Δ (Active - Basal) features for every segment in that file.

See src/events/basal.py for why these are NOT relabeled as a separate
"rest" class in the fixed-window pipeline (that's the dynamic ERD
pipeline's convention, not this one).
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

import config as cfg
from src.features.fractal import compute_all_fractals, compute_hrs_vector
from src.features.spectral import compute_band_powers


@dataclass
class BasalReference:
    """Everything derived from a file's first n_basal segments."""
    n_basal: int
    basal_global: np.ndarray                 # (8,) nanmean over basal segments and channels
    basal_per_ch: Optional[np.ndarray]        # (n_channels, 8) nanmean over basal segments
    basal_ref_segment: Optional[np.ndarray]   # (n_channels, window) nanmean over basal segments
    basal_band_powers: Optional[dict]         # spectral band powers of the basal reference
    basal_martinez: dict                      # {"HO":..., "HV":..., "HRS_p2":..., ...}


def build_basal_reference(segments: List[np.ndarray], n_basal: int = 5,
                           fs: float = None) -> BasalReference:
    """Compute the Δ-reference (basal) features from a file's first segments.

    Args:
        segments: List of (n_channels, window) preprocessed EEG segments for
            one file, in temporal order (as returned by
            src.preprocessing.segmentation.segment_fixed_window).
        n_basal: Number of leading segments to average as the basal
            reference. Clamped to len(segments) if the file is shorter.
        fs: Sampling rate.

    Returns:
        BasalReference. If segments is empty, all fields are NaN/None as
        appropriate (mirrors the original inline behavior).
    """
    fs = fs or cfg.FS
    n_basal = min(n_basal, len(segments))
    basal_segments = segments[:n_basal]

    basal_global = np.full(8, np.nan)
    basal_per_ch = None
    basal_ref_segment = None

    if n_basal > 0:
        n_channels = basal_segments[0].shape[0]
        basal_features_all = np.zeros((n_basal, n_channels, 8))
        for b in range(n_basal):
            basal_features_all[b] = compute_all_fractals(basal_segments[b], fs=fs)
        basal_global = np.nanmean(basal_features_all, axis=(0, 1))
        basal_per_ch = np.nanmean(basal_features_all[..., :8], axis=0)
        basal_ref_segment = np.nanmean(np.stack(basal_segments), axis=0)

    # Basal spectral band powers: computed once here (not per active segment
    # — see compute_spectral_features' basal_band_powers parameter).
    basal_band_powers = None
    if basal_ref_segment is not None:
        basal_band_powers = compute_band_powers(np.mean(basal_ref_segment, axis=0), fs=fs)

    basal_martinez = {}
    if not np.all(np.isnan(basal_global)):
        basal_martinez = {"HO": float(basal_global[5]), "HV": float(basal_global[7])}
        if basal_ref_segment is not None:
            hrs_basal = compute_hrs_vector(np.nanmean(basal_ref_segment, axis=0), fs=fs)
            basal_martinez.update(hrs_basal)
    else:
        basal_martinez = {"HO": np.nan, "HV": np.nan}
        for p in cfg.HRS_PARTITION_VALUES:
            basal_martinez["HRS_p{}".format(p)] = np.nan

    return BasalReference(
        n_basal=n_basal,
        basal_global=basal_global,
        basal_per_ch=basal_per_ch,
        basal_ref_segment=basal_ref_segment,
        basal_band_powers=basal_band_powers,
        basal_martinez=basal_martinez,
    )
