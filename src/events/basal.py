"""
Basal (resting-state) epoch extraction for computing baseline fractal features.

Extracts segments from periods of no-task activity:
1. Gaps between events > MIN_REST_GAP_SEC
2. Start of recording (if no gaps found)
"""
import numpy as np

import config as cfg


def extract_basal_epochs(eeg: np.ndarray, event_indices: np.ndarray,
                          fs: float = None) -> np.ndarray:
    """Extract basal (rest) epochs from inter-event gaps.

    Args:
        eeg: (n_channels, n_samples) preprocessed and bandpass-filtered EEG.
        event_indices: Sample indices of detected events.
        fs: Sampling rate.

    Returns:
        (n_basal_epochs, n_channels, segment_samples) array of basal segments,
        or empty array if no valid basal epochs found.
    """
    fs = fs or cfg.FS
    n_samples = eeg.shape[1]
    segment_samples = round(cfg.BASAL_SEGMENT_DURATION_SEC * fs)
    gap_min = cfg.MIN_REST_GAP_SEC
    gap_pre = cfg.PRE_EVENT_MARGIN_SEC
    max_per_gap = cfg.MAX_BASAL_PER_GAP

    # Convert events to seconds for gap calculation
    t_events = event_indices / fs

    # Build interval edges: [0, event_1, event_2, ..., event_N, end]
    borders = np.concatenate([[0], t_events, [n_samples / fs]])
    intervals = np.diff(borders)

    segments = []

    for i, gap in enumerate(intervals):
        if gap <= gap_min:
            continue

        # Compute rest period boundaries for this interval
        if i == 0:
            # Before first event
            start_rest = 0
            end_rest = int(t_events[0] * fs) - int(gap_pre * fs)
        elif i == len(intervals) - 1:
            # After last event
            start_rest = int(t_events[-1] * fs) + int(cfg.MIN_EVENT_DISTANCE_SEC * fs)
            end_rest = n_samples
        else:
            # Between two events
            start_rest = int(t_events[i - 1] * fs) + int(cfg.MIN_EVENT_DISTANCE_SEC * fs)
            end_rest = int(t_events[i] * fs) - int(gap_pre * fs)

        # Extract up to max_per_gap segments from this rest period
        duration = end_rest - start_rest
        n_possible = duration // segment_samples
        for j in range(min(n_possible, max_per_gap)):
            seg_start = start_rest + j * segment_samples
            seg_end = seg_start + segment_samples
            if seg_start >= 0 and seg_end <= n_samples:
                segments.append(eeg[:, seg_start:seg_end])

    # Fallback — use start of recording
    if len(segments) == 0:
        if n_samples > segment_samples:
            segments.append(eeg[:, :segment_samples])
        else:
            segments.append(eeg)

    return np.stack(segments, axis=0) if segments else np.empty((0,))


def compute_basal_global(basal_epochs: np.ndarray,
                          fractal_fn) -> np.ndarray:
    """Compute global basal features averaged across all basal epochs.

    Args:
        basal_epochs: (n_basal, n_channels, segment_samples) array.
        fractal_fn: Function that computes fractal features from a 1D signal.
                    Should return (n_features,) array.

    Returns:
        (n_channels, n_features) array — mean across basal epochs.
        Also works as (n_features,) if averaged over channels later.
    """
    if basal_epochs.size == 0:
        return np.array([])

    n_basal, n_channels, _ = basal_epochs.shape
    # Determine number of features by running on one segment
    probe = fractal_fn(basal_epochs[0, 0, :])
    n_features = len(probe)

    all_features = np.zeros((n_basal, n_channels, n_features))
    for b in range(n_basal):
        for ch in range(n_channels):
            all_features[b, ch, :] = fractal_fn(basal_epochs[b, ch, :])

    return np.mean(all_features, axis=0)  # (n_channels, n_features)
