"""
Multi-window stability validation for fractal features.

For each detected event, computes all 4 fractal methods at multiple
temporal windows (2s, 3s, 4s, 5s) centered on the event.
Reports:
- Primary value: result from the largest valid window
- Coefficient of variation (CV) across successful windows
- Stability flag: stable / marginal / unstable
"""
import numpy as np

import config as cfg
from src.features.fractal import compute_all_fractals


def compute_stable_features(eeg: np.ndarray, event_center: int,
                             fs: float = None) -> dict:
    """Compute fractal features with stability validation for one event.

    Args:
        eeg: (n_channels, n_samples) preprocessed + filtered EEG.
        event_center: Sample index of the event center.
        fs: Sampling rate.

    Returns:
        Dict with keys for each method:
            {method}_value, {method}_window_sec, {method}_cv, {method}_stability
        plus the same for average.
    """
    fs = fs or cfg.FS
    n_samples = eeg.shape[1]

    methods = ["RS", "Higuchi", "DFA", "Variogram", "Average"]
    method_results = {m: [] for m in methods}
    used_windows = []  # track which window sizes actually fit

    for win_sec in cfg.STABILITY_WINDOWS_SEC:
        win_samples = int(win_sec * fs)
        start = event_center - win_samples // 2
        end = start + win_samples

        if start < 0 or end > n_samples:
            continue

        # Spatial average first (single signal), then compute fractals
        # Much faster than per-channel fractal computation
        segment = eeg[:, start:end]                       # (n_channels, win_samples)
        spatial_mean = np.nanmean(segment, axis=0)        # (win_samples,)
        features = compute_all_fractals(spatial_mean[None, :], fs=fs)  # (1, 5)
        channel_mean = features[0]                         # (5,)

        for i, method in enumerate(methods):
            method_results[method].append(channel_mean[i])

        used_windows.append(win_sec)

    output = {}
    for method in methods:
        values = np.array(method_results[method])
        valid = ~np.isnan(values)

        if np.sum(valid) < 2:
            output[f"{method}_value"] = np.nan
            output[f"{method}_window_sec"] = 0.0
            output[f"{method}_cv"] = np.nan
            output[f"{method}_stability"] = "no_data"
            continue

        valid_values = values[valid]
        mean_val = np.nanmean(valid_values)
        std_val = np.nanstd(valid_values)

        if abs(mean_val) < 1e-10:
            cv = np.nan
            stability = "no_data"
        else:
            cv = std_val / abs(mean_val)
            if cv < cfg.CV_STABLE_THRESHOLD:
                stability = "stable"
            elif cv < cfg.CV_MARGINAL_THRESHOLD:
                stability = "marginal"
            else:
                stability = "unstable"

        # Primary value = largest window's value (most informative)
        primary = valid_values[-1]

        # Longest window actually used
        valid_windows_arr = np.array(used_windows)[valid]
        window_used = valid_windows_arr[-1]

        output[f"{method}_value"] = float(primary)
        output[f"{method}_window_sec"] = float(window_used)
        output[f"{method}_cv"] = float(cv) if not np.isnan(cv) else np.nan
        output[f"{method}_stability"] = stability

    return output


def get_stability_summary(stable_features: dict) -> str:
    """Return a short summary string for stability flags."""
    flags = []
    for method in ["RS", "Higuchi", "DFA", "Variogram", "Average"]:
        flag = stable_features.get(f"{method}_stability", "?")
        flags.append(f"{method}={flag}")
    return " | ".join(flags)
