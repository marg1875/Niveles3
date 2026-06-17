"""
Event detection for both asymptomatic (motor imagery/ERD) and
synchronous (stimulus-driven/markers) paradigms.
"""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

import config as cfg


def detect_erd_events(eeg_filtered: np.ndarray, data_raw: dict = None,
                       fs: float = None) -> np.ndarray:
    """Detect motor imagery events via ERD in Cz channel (index 5).

    Computes mu/beta band energy, finds negative peaks (desynchronization),
    and applies auto-calibrated threshold.

    Args:
        eeg_filtered: (n_channels, n_samples) preprocessed EEG.
        data_raw: Not used for ERD, but kept for API consistency.
        fs: Sampling rate.

    Returns:
        Array of sample indices where events were detected.
    """
    fs = fs or cfg.FS
    channel = cfg.CHANNEL_ERD  # Cz = index 5
    n_samples = eeg_filtered.shape[1]

    # Extract mu/beta band from Cz
    b, a = butter(4, cfg.MU_BAND, btype="bandpass", fs=fs)
    signal_band = filtfilt(b, a, eeg_filtered[channel])

    # Compute energy (moving average, 1-second window)
    window = fs
    energy = np.convolve(signal_band ** 2, np.ones(window) / window, mode="same")

    # Auto-calibration
    p_min = np.percentile(energy, cfg.ERD_PERCENTILE_MIN)
    p_max = np.percentile(energy, cfg.ERD_PERCENTILE_MAX)
    threshold = p_min + cfg.ERD_THRESHOLD_FACTOR * (p_max - p_min)

    # Find negative peaks (desynchronization = low energy)
    locs, _ = find_peaks(
        -energy,
        height=-threshold,
        distance=int(fs * cfg.MIN_EVENT_DISTANCE_SEC),
        width=int(fs * cfg.MIN_EVENT_WIDTH_SEC),
    )

    # Deduplication — remove events closer than event_distance
    if len(locs) > 1:
        locs = _deduplicate(locs, fs)

    return locs


def detect_stimulus_events(data_raw: dict, eeg_filtered: np.ndarray,
                            fs: float = None) -> np.ndarray:
    """Detect stimulus-driven events from marker vectors.

    Handles three formats:
    1. Binary mask (values are 0/constant pattern) — detect rising edges
    2. Time vector in seconds — convert to sample indices
    3. Sample indices — use directly
    """
    fs = fs or cfg.FS
    n_samples = eeg_filtered.shape[1]

    for field_name in cfg.MARKER_FIELDS:
        if field_name not in data_raw:
            continue
        try:
            stims_raw = np.array(data_raw[field_name], dtype=float).flatten()
        except (ValueError, TypeError):
            continue

        if len(stims_raw) == 0:
            continue

        unique_vals = np.unique(stims_raw)

        # Case 1: Binary mask (only 2-3 unique values, all non-negative)
        if len(unique_vals) <= 3 and np.all(unique_vals >= 0):
            diff_s = np.diff(stims_raw)
            locs = np.where(diff_s > 0)[0] + 1

            if len(locs) > 0:
                # Scale short binary masks to full signal length
                if len(stims_raw) < n_samples:
                    scale = n_samples / len(stims_raw)
                    locs = np.round(locs * scale).astype(int)
                locs = locs[(locs > 0) & (locs < n_samples)]

        elif len(stims_raw) >= n_samples - 100:
            # Binary vector at sample resolution
            diff_s = np.diff(stims_raw)
            locs = np.where(diff_s > 0)[0] + 1

        else:
            # Short vector of times or indices
            is_increasing = np.all(np.diff(stims_raw) >= 0)
            max_val = np.max(stims_raw)

            if is_increasing and max_val > 0:
                if max_val <= (n_samples / fs) + 5:
                    locs = np.round(stims_raw * fs).astype(int)
                else:
                    locs = np.round(stims_raw).astype(int)
            else:
                if np.max(stims_raw) <= (n_samples / fs) + 5:
                    locs = np.round(stims_raw * fs).astype(int)
                else:
                    locs = np.round(stims_raw).astype(int)

        # Filter valid indices
        locs = locs[(locs > 0) & (locs < n_samples) & ~np.isnan(locs)]
        locs = locs.astype(int)

        # Deduplication
        if len(locs) > 1:
            locs = _deduplicate(locs, fs)

        if len(locs) > 0:
            return locs

    return np.array([], dtype=int)


def detect_events(eeg_filtered: np.ndarray, data_raw: dict,
                  is_imagery: bool, fs: float = None) -> np.ndarray:
    """Unified event detection dispatcher.

    Args:
        eeg_filtered: Preprocessed EEG (n_channels, n_samples).
        data_raw: Loaded .mat dict.
        is_imagery: True for motor imagery (ERD), False for stimulus-driven.
        fs: Sampling rate.

    Returns:
        Array of sample indices for detected events.
    """
    if is_imagery:
        return detect_erd_events(eeg_filtered, data_raw, fs=fs)
    else:
        return detect_stimulus_events(data_raw, eeg_filtered, fs=fs)


def _deduplicate(locs: np.ndarray, fs: float) -> np.ndarray:
    """Remove events closer than MIN_EVENT_DISTANCE_SEC."""
    min_dist = int(cfg.MIN_EVENT_DISTANCE_SEC * fs)
    locs_sorted = np.sort(locs)
    dedup = [locs_sorted[0]]
    for ev in locs_sorted[1:]:
        if ev - dedup[-1] >= min_dist:
            dedup.append(ev)
    result = np.array(dedup, dtype=int)
    if len(result) < len(locs):
        pass  # logging could go here
    return result
